from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
from .tokenizer import H4Tokenizer


class ASRDataset(Dataset):
    def __init__(
            self,
            partition: Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config: dict,
            tokenizer: H4Tokenizer,
            isTrainPartition: bool,
            global_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.
        """
        # Store basic configuration
        self.config = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Get tokenizer ids for special tokens
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths
        root = config['root']
        self.fbank_dir = os.path.join(root, partition, 'fbank')

        # Get all feature files in sorted order
        self.fbank_files = sorted([
            f for f in os.listdir(self.fbank_dir) if f.endswith('.npy')
        ])

        # Take subset if specified
        subset_size = config.get('subset_size', None)
        if subset_size is not None:
            self.fbank_files = self.fbank_files[:subset_size]

        # Number of samples
        self.length = len(self.fbank_files)

        # Handle text files for non-test partitions
        if self.partition != "test-clean":
            self.text_dir = os.path.join(root, partition, 'text')
            self.text_files = sorted([
                f for f in os.listdir(self.text_dir) if f.endswith('.npy')
            ])
            if subset_size is not None:
                self.text_files = self.text_files[:subset_size]

            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")

        # Initialize storage lists
        self.feats = []
        self.transcripts_shifted = []
        self.transcripts_golden  = []

        # Counters
        self.total_chars  = 0
        self.total_tokens = 0

        # Max length variables
        self.feat_max_len = 0
        self.text_max_len = 0

        # Welford accumulators for global_mvn
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2   = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load features — shape is (num_feats, time), e.g. (80, T)
            feat = np.load(os.path.join(self.fbank_dir, self.fbank_files[i]))

            # feat shape: (num_feats, time) — truncate feature dim if needed
            feat = feat[:self.config['num_feats'], :]  # (num_feats, time)

            # Store as float32 tensor
            feat = torch.FloatTensor(feat)  # (num_feats, time)

            self.feats.append(feat)

            # Track max length (time dimension)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update Welford accumulators
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                batch_count = feat.shape[1]
                count += batch_count
                delta  = feat - mean.unsqueeze(1)
                mean  += delta.mean(dim=1)
                delta2 = feat - mean.unsqueeze(1)
                M2    += (delta * delta2).sum(dim=1)

            if self.partition != "test-clean":
                # Load transcript — numpy array of individual characters, e.g. ['H','I',' ','T','H','E','R','E']
                transcript_arr = np.load(os.path.join(self.text_dir, self.text_files[i]), allow_pickle=True)
                # Join characters into a single string
                transcript = ''.join(transcript_arr.tolist())
                transcript = transcript.strip()

                # Track character count
                self.total_chars += len(transcript)

                # Tokenize
                tokenized = tokenizer.encode(transcript)

                # Track token count
                self.total_tokens += len(tokenized)

                # Track max length
                self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

                # Create shifted (SOS + tokens) and golden (tokens + EOS) versions
                shifted = torch.LongTensor([self.sos_token] + tokenized)
                golden  = torch.LongTensor(tokenized + [self.eos_token])
                self.transcripts_shifted.append(shifted)
                self.transcripts_golden.append(golden)

        # Average chars per token
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        if self.partition != "test-clean":
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # Compute global stats
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                variance = M2 / (count - 1)
                self.global_std  = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        '''Get average number of characters per token.'''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        Returns:
            feat: FloatTensor of shape (num_feats, time)
            shifted_transcript: LongTensor or None
            golden_transcript: LongTensor or None
        """
        feat = self.feats[idx]  # (num_feats, time)

        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            assert self.global_mean is not None and self.global_std is not None
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass

        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = self.transcripts_shifted[idx]
            golden_transcript  = self.transcripts_golden[idx]

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch.
        Returns:
            padded_feats:  (batch, max_time, num_feats)
            padded_shifted: (batch, max_len) or None
            padded_golden:  (batch, max_len) or None
            feat_lengths:   (batch,)
            transcript_lengths: (batch,) or None
        """
        # Collect features transposed to (time, num_feats) for pad_sequence
        batch_feats = [item[0].T for item in batch]   # list of (time, num_feats)

        # Feature lengths
        feat_lengths = torch.LongTensor([f.shape[0] for f in batch_feats])

        # Pad features: pad_sequence expects list of (T, F), returns (max_T, B, F) -> transpose to (B, max_T, F)
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)  # (B, max_T, F)

        # Handle transcripts
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            batch_shifted = [item[1] for item in batch]
            batch_golden  = [item[2] for item in batch]

            transcript_lengths = torch.LongTensor([len(s) for s in batch_shifted])

            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)  # (B, max_len)
            padded_golden  = pad_sequence(batch_golden,  batch_first=True, padding_value=self.pad_token)  # (B, max_len)

        # Apply SpecAugment (training only)
        if self.config["specaug"] and self.isTrainPartition:
            # Permute to (B, F, T) for SpecAugment
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, F, T)

            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            # Permute back to (B, T, F)
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, T, F)

        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths
