from typing import Tuple, List
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
try:
    from .tokenizer import H4Tokenizer
except ImportError:
    import importlib.util as _ilu, sys as _sys, os as _os
    _tok_path = _os.path.join(_os.path.dirname(__file__), 'tokenizer.py')
    _tok_spec = _ilu.spec_from_file_location('hw4lib.data.tokenizer', _tok_path)
    _tok_mod  = _ilu.module_from_spec(_tok_spec)
    _sys.modules['hw4lib.data.tokenizer'] = _tok_mod
    _tok_spec.loader.exec_module(_tok_mod)
    H4Tokenizer = _tok_mod.H4Tokenizer


class LMDataset(Dataset):
    """
    Dataset for Language Model training/evaluation.
    """
    def __init__(
            self,
            partition: str,
            config: dict,
            tokenizer: H4Tokenizer
    ):
        # Store configuration and other args
        # DO NOT MODIFY
        self.config    = config
        self.partition = partition
        self.tokenizer = tokenizer

        # Special token ids
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths: config['root'] already points to hw4p1_data directory
        # (e.g. "/local/hw4_data/hw4p1_data"), so just append partition
        self.text_dir = os.path.join(config['root'], partition)

        # All .npy files in sorted order
        self.text_files = sorted([
            os.path.join(self.text_dir, f)
            for f in os.listdir(self.text_dir)
            if f.endswith('.npy')
        ])

        # Take subset if configured
        subset_size = config.get('subset', None)
        if subset_size is not None:
            self.text_files = self.text_files[:subset_size]

        # Initialize lists to store transcripts
        self.transcripts_shifted = []
        self.transcripts_golden  = []

        # Initialize tracking variables
        # DO NOT MODIFY
        self.total_chars  = 0
        self.total_tokens = 0
        self.text_max_len = 0

        print(f"Loading transcripts for {partition} partition...")
        for file in tqdm(self.text_files):
            # Load transcript string from .npy file
            transcript = str(np.load(file, allow_pickle=True).item())

            # Track character count (before tokenization)
            # DO NOT MODIFY
            self.total_chars += len(transcript)

            # Tokenize the transcript (returns list of token ids)
            tokenized = tokenizer.encode(transcript)

            # Track token count (excluding special tokens)
            # DO NOT MODIFY
            self.total_tokens += len(tokenized)

            # Track max length (add 1 for sos/eos)
            # DO NOT MODIFY
            self.text_max_len = max(self.text_max_len, len(tokenized) + 1)

            # Shifted: [SOS] + tokenized  (input to model)
            # Golden:  tokenized + [EOS]  (prediction targets)
            self.transcripts_shifted.append([self.sos_token] + tokenized)
            self.transcripts_golden.append(tokenized + [self.eos_token])

        # Calculate average characters per token
        # DO NOT MODIFY
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0

        # Verify data alignment
        # DO NOT MODIFY
        if not (len(self.transcripts_shifted) == len(self.transcripts_golden)):
            raise ValueError("Shifted and golden transcripts are misaligned")

        # Store the length of the dataset
        self.length = len(self.transcripts_shifted)

    def get_avg_chars_per_token(self) -> float:
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        shifted = torch.LongTensor(self.transcripts_shifted[idx])
        golden  = torch.LongTensor(self.transcripts_golden[idx])
        return shifted, golden

    def collate_fn(self, batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Unzip into separate lists
        shifted_transcripts, golden_transcripts = zip(*batch)

        # Record sequence lengths before padding (shifted and golden have same length)
        lengths = torch.LongTensor([len(s) for s in shifted_transcripts])

        # Pad sequences to the longest in the batch
        padded_shifted = pad_sequence(shifted_transcripts, batch_first=True, padding_value=self.pad_token)
        padded_golden  = pad_sequence(golden_transcripts,  batch_first=True, padding_value=self.pad_token)

        return padded_shifted, padded_golden, lengths

    def sample_prompts(self, num_samples: int, prompt_length: int, seed: int = None) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        """
        Sample random prompts of fixed length from the dataset and return their original sequences.
        DO NOT MODIFY
        """
        if seed is not None:
            np_state = np.random.get_state()
            np.random.seed(seed)

        prompts   = []
        originals = []
        attempts  = 0
        max_attempts = num_samples * 10

        while len(prompts) < num_samples and attempts < max_attempts:
            idx    = np.random.randint(0, len(self))
            tokens = self.transcripts_shifted[idx][1:]  # remove sos token

            if len(tokens) < prompt_length:
                attempts += 1
                continue

            prompt_tokens = tokens[:prompt_length]
            prompts.append(torch.LongTensor([self.sos_token] + prompt_tokens))
            originals.append(torch.LongTensor(tokens + [self.eos_token]))
            attempts += 1

        if len(prompts) < num_samples:
            print(f"Warning: Could only sample {len(prompts)} valid prompts")

        if seed is not None:
            np.random.set_state(np_state)

        return torch.stack(prompts), originals
