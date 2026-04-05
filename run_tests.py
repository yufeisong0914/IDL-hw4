"""Quick test runner for HW4P1 implementations."""
import sys, importlib
sys.path.insert(0, '.')

# ── MyTorch (no torch dependencies) ────────────────────────────────────────
from tests.test_mytorch_linear import test_linear
from tests.test_mytorch_softmax import test_softmax
from tests.test_mytorch_scaled_dot_product_attention import test_scaled_dot_product_attention
from tests.test_mytorch_multi_head_attention import test_multi_head_attention

test_linear(); print("[PASS] Linear")
test_softmax(); print("[PASS] Softmax")
test_scaled_dot_product_attention(); print("[PASS] ScaledDotProductAttention")
test_multi_head_attention(); print("[PASS] MultiHeadAttention")

# ── hw4lib model (import modules directly, bypass __init__) ─────────────────
# Import model modules directly to avoid hw4lib/__init__.py chain
import importlib.util as ilu

def load_module(path, name):
    spec = ilu.spec_from_file_location(name, path)
    mod = ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

import torch, torch.nn as nn

# Load mask modules
masks_mod = load_module('hw4lib/model/masks.py', 'hw4lib.model.masks')
CausalMask = masks_mod.CausalMask
PadMask    = masks_mod.PadMask

pe_mod = load_module('hw4lib/model/positional_encoding.py', 'hw4lib.model.positional_encoding')
PositionalEncoding = pe_mod.PositionalEncoding

# Load sublayers (depends on torch.nn only)
sublayers_mod = load_module('hw4lib/model/sublayers.py', 'hw4lib.model.sublayers')
SelfAttentionLayer  = sublayers_mod.SelfAttentionLayer
CrossAttentionLayer = sublayers_mod.CrossAttentionLayer
FeedForwardLayer    = sublayers_mod.FeedForwardLayer

# Load encoder/decoder layers
enc_layers_mod = load_module('hw4lib/model/encoder_layers.py', 'hw4lib.model.encoder_layers')
SelfAttentionEncoderLayer = enc_layers_mod.SelfAttentionEncoderLayer

dec_layers_mod = load_module('hw4lib/model/decoder_layers.py', 'hw4lib.model.decoder_layers')
SelfAttentionDecoderLayer  = dec_layers_mod.SelfAttentionDecoderLayer
CrossAttentionDecoderLayer = dec_layers_mod.CrossAttentionDecoderLayer

# Mock torchaudio and stub out hw4lib package __init__ files to avoid import chain issues
import types
for mod_name in ['torchaudio', 'torchaudio.transforms', 'torchaudio.functional']:
    m = types.ModuleType(mod_name)
    m.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
    sys.modules[mod_name] = m

# Stub hw4lib and hw4lib.model packages so their __init__.py never runs
# (hw4lib/__init__.py would pull in asr_dataset -> torchaudio, etc.)
for pkg in ['hw4lib', 'hw4lib.model', 'hw4lib.data', 'hw4lib.trainers', 'hw4lib.utils', 'hw4lib.decoding']:
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__spec__ = importlib.machinery.ModuleSpec(pkg, None)
        m.__path__ = []
        m.__package__ = pkg
        sys.modules[pkg] = m

speech_emb_mod = load_module('hw4lib/model/speech_embedding.py', 'hw4lib.model.speech_embedding')
transformers_mod = load_module('hw4lib/model/transformers.py', 'hw4lib.model.transformers')
DecoderOnlyTransformer = transformers_mod.DecoderOnlyTransformer

# ── Run model tests ────────────────────────────────────────────────────────
from tests.test_mask_causal import test_mask_causal
from tests.test_mask_padding import test_mask_padding
test_mask_causal(CausalMask); print("[PASS] CausalMask")
test_mask_padding(PadMask); print("[PASS] PadMask")

from tests.test_positional_encoding import test_positional_encoding
test_positional_encoding(PositionalEncoding); print("[PASS] PositionalEncoding")

from tests.test_sublayer_selfattention import test_sublayer_selfattention
from tests.test_sublayer_feedforward import test_sublayer_feedforward
from tests.test_sublayer_crossattention import test_sublayer_crossattention
test_sublayer_selfattention(SelfAttentionLayer); print("[PASS] SelfAttentionLayer")
test_sublayer_feedforward(FeedForwardLayer); print("[PASS] FeedForwardLayer")
test_sublayer_crossattention(CrossAttentionLayer); print("[PASS] CrossAttentionLayer")


from tests.test_decoderlayer_selfattention import test_decoderlayer_selfattention
test_decoderlayer_selfattention(SelfAttentionDecoderLayer); print("[PASS] SelfAttentionDecoderLayer")
from tests.test_decoderlayer_crossattention import test_cross_attention_decoder_layer
test_cross_attention_decoder_layer(CrossAttentionDecoderLayer); print("[PASS] CrossAttentionDecoderLayer")

from tests.test_encoderlayer_selfattention import test_self_attention_encoder_layer
test_self_attention_encoder_layer(SelfAttentionEncoderLayer); print("[PASS] SelfAttentionEncoderLayer")

from tests.test_transformer_decoder_only import test_decoder_only_transformer
test_decoder_only_transformer(DecoderOnlyTransformer); print("[PASS] DecoderOnlyTransformer")



# ── Dataset ────────────────────────────────────────────────────────────────
from tests.test_dataset_lm import test_lm_dataset
lm_dataset_mod = load_module('hw4lib/data/lm_dataset.py', 'hw4lib.data.lm_dataset')
tokenizer_mod  = load_module('hw4lib/data/tokenizer.py',  'hw4lib.data.tokenizer')
LMDataset   = lm_dataset_mod.LMDataset
H4Tokenizer = tokenizer_mod.H4Tokenizer
test_lm_dataset(LMDataset, H4Tokenizer); print("[PASS] LMDataset")

# ── Decoding ───────────────────────────────────────────────────────────────
from tests.test_decoding import test_greedy_decoding
test_greedy_decoding(); print("[PASS] Greedy Decoding")

print("\n=== All tests passed! ===")
