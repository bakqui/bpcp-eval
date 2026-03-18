import sys
import os
import contextlib
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoConfig, AutoModel

def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)

# =========================
# 1) Continuous-time RoPE
# =========================
def _build_base_frequencies(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    assert head_dim % 2 == 0, "head_dim must be even for rotary."
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return base ** (-idx / head_dim)

def ct_rope_angles(
    times: torch.Tensor,               # [B, T] continuous timestamps
    head_dim: int,
    base: float = 10000.0,
    learnable_scale: Optional[nn.Parameter] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns cos, sin of shape [B, 1, T, head_dim]
    """
    B, T = times.shape
    freqs = _build_base_frequencies(head_dim, base).to(times.device)  # [Hd/2]
    if learnable_scale is not None:
        freqs = freqs * learnable_scale

    theta = times.unsqueeze(-1) * freqs.view(1, 1, -1)  # [B, T, Hd/2]
    cos = torch.zeros(B, T, head_dim, device=times.device, dtype=torch.float32)
    sin = torch.zeros_like(cos)
    cos[..., 0::2] = torch.cos(theta)
    cos[..., 1::2] = cos[..., 0::2]
    sin[..., 0::2] = torch.sin(theta)
    sin[..., 1::2] = sin[..., 0::2]
    return cos.unsqueeze(1), sin.unsqueeze(1)  # [B, 1, T, Hd]

# ==============================================
# 2) Qwen3-compatible CT-RoPE (top-level module)
# ==============================================

class CTRotaryEmbeddingQwen3(nn.Module):
    """
    Drop-in replacement for Qwen3's model.rotary_emb.

    Qwen3 calls:
        cos, sin = self.rotary_emb(hidden_states, position_ids)
    and expects cos/sin shaped [B, T, head_dim] (then unsqueezed inside attention).
    """
    def __init__(self, head_dim: int, base: float = 10000.0, learnable: bool = True):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scale = nn.Parameter(torch.ones(head_dim // 2)) if learnable else None
        self._ctx_times = None  # set via context manager

    def set_times(self, times: torch.Tensor):
        # times: [B, T]
        self._ctx_times = times

    def clear_times(self):
        self._ctx_times = None

    def forward(self, hidden_states: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """
        Return cos, sin shaped [B, T, head_dim].
        """
        B, T, _ = hidden_states.shape
        device = hidden_states.device

        if self._ctx_times is None:
            times = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)  # [1, T]
        else:
            times = self._ctx_times.to(device)  # [B, T]

        cos, sin = ct_rope_angles(times, self.head_dim, base=self.base, learnable_scale=self.scale)  # [B,1,T,Hd]
        cos, sin = cos.squeeze(1), sin.squeeze(1)  # -> [B, T, Hd]
        assert cos.size(-1) == self.head_dim
        return cos, sin

# ==============================
# 3) Context manager for times
# ==============================
@contextlib.contextmanager
def ct_rope_ctx(model: nn.Module, times: torch.Tensor):
    keep = []
    try:
        for _, m in model.named_modules():
            if hasattr(m, "rotary_emb") and isinstance(m.rotary_emb, CTRotaryEmbeddingQwen3):
                m.rotary_emb.set_times(times)
                keep.append(m.rotary_emb)
        yield
    finally:
        for r in keep:
            r.clear_times()

def patch_qwen3_model_rotary(hf_model: nn.Module):
    """
    Replace Qwen3's rotary_emb with CTRotaryEmbeddingQwen3 using:
        head_dim = hidden_size // num_key_value_heads
    This matches HF attention’s RoPE expectation (cos/sin [B, T, Hd], unsqueeze inside).
    """
    cfg = hf_model.config
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_kv is None:
        n_kv = getattr(cfg, "num_attention_heads", None)
    if n_kv is None:
        raise ValueError("Config missing num_key_value_heads/num_attention_heads")
    head_dim = cfg.hidden_size // n_kv
    rope_theta = getattr(cfg, "rope_theta", 10000.0)

    # Most Qwen3 models expose a top-level rotary_emb
    if hasattr(hf_model, "rotary_emb"):
        hf_model.rotary_emb = CTRotaryEmbeddingQwen3(head_dim, rope_theta, True)
        return
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "rotary_emb"):
        hf_model.model.rotary_emb = CTRotaryEmbeddingQwen3(head_dim, rope_theta, True)
        return

    # Fallback: search & replace any submodule named 'rotary_emb'
    for name, _ in hf_model.named_modules():
        if name.endswith("rotary_emb"):
            parent = hf_model
            for p in name.split(".")[:-1]:
                parent = getattr(parent, p)
            setattr(parent, name.split(".")[-1], CTRotaryEmbeddingQwen3(head_dim, rope_theta, True))
            return

    raise ValueError("No rotary_emb module found in the Qwen3 model.")

def build_ct_rope_qwen3_small(params, hidden_size=256, heads=4, kv_heads=4, inter_size=256):
    """
    Keep Qwen3 “as-is” but small: head_dim becomes 128 (compatible), since 256 / 2 = 128.
    """
    cfg_name = "Qwen/Qwen3-8B"
    config = AutoConfig.from_pretrained(cfg_name)

    # small, Qwen-consistent config
    config.num_hidden_layers = 4
    config.hidden_size = hidden_size           # 256
    config.num_attention_heads = heads         # 2
    config.num_key_value_heads = kv_heads      # 2  (MHA; head_dim = 128)
    config.intermediate_size = inter_size      # 512 or 4x if you prefer
    config.vocab_size = 1                      # unused for time-series

    base_model = AutoModel.from_config(config) # Qwen3Model, not CausalLM

    # Patch rotary to CT-RoPE using head_dim = hidden_size // num_key_value_heads
    # patch_qwen3_model_rotary(base_model)

    base_model.to(torch.float32).eval()

    model = import_from(f'src.models.{params.model_name}', params.model_name)
    return model(base_model, input_dim=2, output_dim=2).eval()

def build_model(params, logger):

    model = import_from(f'src.models.{params.model_name}', params.model_name)

    if params.model_name in ['TimeSeriesQwen3', 'TimeSeriesQwen3Stat']:
        model = build_ct_rope_qwen3_small(params)
    else:
        model = model(params)

    if os.path.isfile(params.reload_path):
        reloaded_model = torch.load(params.reload_path)
        model.load_state_dict(reloaded_model['model'])
        logger.info("========Reloaded model from {}".format(params.reload_path))

    return model
