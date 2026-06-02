"""Shifted-window self-attention diffusion transformer for RVR.

Adapted from the ``WanTransformer3DModel`` (Wan2.2-TI2V) implementation in
Hugging Face ``diffusers`` (Apache-2.0). The mask-free shifted-window
self-attention processor and the multi-backend dense-attention dispatcher are
specific to RVR; the same checkpoint runs bit-identically across PyTorch SDPA,
FlashAttention-2/3, SageAttention and xFormers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

logger = logging.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Attention backends                                                          #
# --------------------------------------------------------------------------- #

_AVAILABLE_BACKENDS: set = {"sdpa"}
_BACKEND_PRIORITY = ("flash_attn_3", "flash_attn_2", "sageattention", "sdpa", "xformers")
_ATTN_BACKEND: Optional[str] = None

try:
    from flash_attn_interface import flash_attn_func as _fa3_func
    _AVAILABLE_BACKENDS.add("flash_attn_3")
except Exception:
    _fa3_func = None

try:
    from flash_attn import flash_attn_func as _fa2_func
    _AVAILABLE_BACKENDS.add("flash_attn_2")
except Exception:
    _fa2_func = None

try:
    from sageattention import sageattn as _sage_func
    _AVAILABLE_BACKENDS.add("sageattention")
except Exception:
    _sage_func = None

try:
    import xformers.ops as _xops
    _xformers_mea = _xops.memory_efficient_attention
    _AVAILABLE_BACKENDS.add("xformers")
except Exception:
    _xformers_mea = None


def list_available_attention_backends() -> Tuple[str, ...]:
    return tuple(sorted(_AVAILABLE_BACKENDS))


def _pick_best_backend() -> str:
    for name in _BACKEND_PRIORITY:
        if name in _AVAILABLE_BACKENDS:
            return name
    return "sdpa"


def set_attention_backend(name: str = "auto") -> str:
    global _ATTN_BACKEND
    if name == "auto":
        _ATTN_BACKEND = _pick_best_backend()
    elif name not in _AVAILABLE_BACKENDS:
        raise ValueError(
            f"Attention backend {name!r} is not available. "
            f"Available backends: {list_available_attention_backends()}"
        )
    else:
        _ATTN_BACKEND = name
    logger.info(f"[rvr] attention backend = {_ATTN_BACKEND!r}")
    return _ATTN_BACKEND


def get_attention_backend() -> str:
    global _ATTN_BACKEND
    if _ATTN_BACKEND is None:
        _ATTN_BACKEND = _pick_best_backend()
    return _ATTN_BACKEND


def _dense_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Mask-free dense attention. Inputs/outputs are (B, N, H, D)."""
    backend = get_attention_backend()

    if backend == "flash_attn_3" and _fa3_func is not None:
        out = _fa3_func(q, k, v, causal=False)
        return out[0] if isinstance(out, tuple) else out

    if backend == "flash_attn_2" and _fa2_func is not None:
        return _fa2_func(q, k, v, dropout_p=0.0, causal=False)

    if backend == "sageattention" and _sage_func is not None:
        out = _sage_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                         tensor_layout="HND", is_causal=False)
        return out.transpose(1, 2)

    if backend == "xformers" and _xformers_mea is not None:
        return _xformers_mea(q, k, v, attn_bias=None)

    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


# --------------------------------------------------------------------------- #
# Window partition caches                                                     #
# --------------------------------------------------------------------------- #

def _make_hw_starts(H, W, wh, ww, do_shift, device=None):
    device = device or torch.device("cpu")

    def _axis_starts(size: int, win: int) -> torch.Tensor:
        if size <= win:
            return torch.zeros(1, dtype=torch.long, device=device)
        shift = (win // 2) if do_shift else 0
        max_start = size - win
        n = (size + win - 1) // win + 2
        k = torch.arange(n, dtype=torch.long, device=device)
        starts = (k * win - shift).clamp_(0, max_start)
        starts = torch.unique(starts, sorted=True)
        if starts.numel() > 2:
            prev, nxt = starts[:-2], starts[2:]
            covered = nxt <= (prev + win)
            keep = torch.ones_like(starts, dtype=torch.bool)
            keep[1:-1] = ~covered
            starts = starts[keep]
        return starts

    return _axis_starts(H, wh), _axis_starts(W, ww)


def _build_hw_lin_indices(T, H, W, h_starts, w_starts, wh, ww):
    device = h_starts.device
    dh = torch.arange(wh, device=device)
    dw = torch.arange(ww, device=device)
    dt = torch.arange(T, device=device)
    h_idx = h_starts[:, None] + dh[None, :]
    w_idx = w_starts[:, None] + dw[None, :]
    spatial_lin = (h_idx[:, None, :, None] * W + w_idx[None, :, None, :]).reshape(-1, wh * ww)
    full_lin = dt[None, :, None] * (H * W) + spatial_lin[:, None, :]
    return full_lin.reshape(spatial_lin.shape[0], T * wh * ww)


class _WindowIndexCache:
    _store: Dict[tuple, torch.Tensor] = {}

    @classmethod
    def get(cls, T, H, W, wh, ww, do_shift, device):
        key = (T, H, W, wh, ww, do_shift, device.type, device.index)
        if key not in cls._store:
            h_s, w_s = _make_hw_starts(H, W, wh, ww, do_shift, device)
            cls._store[key] = _build_hw_lin_indices(T, H, W, h_s, w_s, wh, ww)
        return cls._store[key]

    @classmethod
    def clear(cls):
        cls._store.clear()


class _WindowRuntimeMeta:
    __slots__ = ("lin_flat", "owner_pos", "Nw", "Lw", "THW")

    def __init__(self, lin_flat, owner_pos, Nw, Lw, THW):
        self.lin_flat = lin_flat
        self.owner_pos = owner_pos
        self.Nw = Nw
        self.Lw = Lw
        self.THW = THW


class _WindowRuntimeMetaCache:
    _store: Dict[tuple, _WindowRuntimeMeta] = {}

    @staticmethod
    def _build_owner_pos_cpu(lin, prefer_front, THW):
        lin_cpu = lin.detach().to("cpu")
        Nw, Lw = lin_cpu.shape
        owner = torch.empty(THW, dtype=torch.long)
        local = torch.arange(Lw, dtype=torch.long)
        order_iter = range(Nw - 1, -1, -1) if prefer_front else range(Nw)
        for wi in order_iter:
            owner[lin_cpu[wi]] = wi * Lw + local
        return owner

    @classmethod
    def get(cls, T, H, W, wh, ww, do_shift, prefer_front, device):
        key = (T, H, W, wh, ww, bool(do_shift), bool(prefer_front), device.type, device.index)
        if key not in cls._store:
            lin = _WindowIndexCache.get(T, H, W, wh, ww, do_shift, device)
            Nw, Lw = lin.shape
            THW = T * H * W
            owner_cpu = cls._build_owner_pos_cpu(lin, prefer_front, THW)
            cls._store[key] = _WindowRuntimeMeta(
                lin_flat=lin.reshape(-1).contiguous(),
                owner_pos=owner_cpu.to(device=device, non_blocking=True),
                Nw=int(Nw), Lw=int(Lw), THW=int(THW))
        return cls._store[key]

    @classmethod
    def clear(cls):
        cls._store.clear()


# --------------------------------------------------------------------------- #
# Rotary embedding helpers                                                    #
# --------------------------------------------------------------------------- #

def _apply_rotary_emb(x, freqs_cos, freqs_sin):
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    if cos.dtype != x.dtype:
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _apply_rotary_emb_inplace(x, freqs_cos, freqs_sin):
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    if cos.dtype != x.dtype:
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x_pair = x.view(*x.shape[:-1], -1, 2)
    x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
    tmp = x_even * sin
    x_even.mul_(cos)
    x_even.addcmul_(x_odd, sin, value=-1)
    x_odd.mul_(cos)
    x_odd.add_(tmp)
    del tmp
    return x


def _release_input_storage(t: torch.Tensor) -> None:
    """Free a tensor's CUDA storage to bypass Python refcount held by *args."""
    try:
        if t.is_cuda and t._base is None and t.is_contiguous():
            t.untyped_storage().resize_(0)
    except Exception:
        pass


def _get_qkv_projections(attn, hidden_states, encoder_hidden_states):
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    if getattr(attn, "fused_projections", False):
        if attn.cross_attention_dim_head is None:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn, encoder_hidden_states_img):
    if getattr(attn, "fused_projections", False):
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


def _infer_local_thw(thw_global, k_local):
    Tg, Hg, Wg = thw_global
    if k_local == Tg * Hg * Wg:
        return Tg, Hg, Wg
    if k_local % (Hg * Wg) == 0:
        return (k_local // (Hg * Wg), Hg, Wg)
    if k_local % (Tg * Wg) == 0:
        return (Tg, k_local // (Tg * Wg), Wg)
    if k_local % (Tg * Hg) == 0:
        return (Tg, Hg, k_local // (Tg * Hg))
    raise RuntimeError(f"Cannot infer local THW from {thw_global} and k_local={k_local}.")


# --------------------------------------------------------------------------- #
# Attention modules                                                           #
# --------------------------------------------------------------------------- #

class WanAttnProcessor:
    """Standard global attention used for cross-attention."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor requires PyTorch 2.0+.")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        del hidden_states

        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img).unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = dispatch_attention_fn(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend, parallel_config=self._parallel_config)
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=self._attention_backend, parallel_config=self._parallel_config)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        if attn.training and isinstance(attn.to_out[1], nn.Dropout) and attn.to_out[1].p > 0.0:
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(self, dim, heads=8, dim_head=64, eps=1e-5, dropout=0.0,
                 added_kv_proj_dim=None, cross_attention_dim_head=None,
                 processor=None, is_cross_attention=None):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (self.inner_dim if cross_attention_dim_head is None
                             else cross_attention_dim_head * heads)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout)])
        self.norm_q = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = nn.RMSNorm(dim_head * heads, eps=eps)

        self.is_cross_attention = cross_attention_dim_head is not None
        self.fused_projections = False
        self.set_processor(processor)

    def fuse_projections(self):
        if self.fused_projections:
            return

        if self.cross_attention_dim_head is None:
            w = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            b = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_f, in_f = w.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_f, out_f, bias=True)
            self.to_qkv.load_state_dict({"weight": w, "bias": b}, strict=True, assign=True)
        else:
            w = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            b = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_f, in_f = w.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_f, out_f, bias=True)
            self.to_kv.load_state_dict({"weight": w, "bias": b}, strict=True, assign=True)

        if self.added_kv_proj_dim is not None:
            w = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            b = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_f, in_f = w.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_f, out_f, bias=True)
            self.to_added_kv.load_state_dict({"weight": w, "bias": b}, strict=True, assign=True)

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        for attr in ("to_qkv", "to_kv", "to_added_kv"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.fused_projections = False

    def forward(self, hidden_states, encoder_hidden_states=None,
                attention_mask=None, rotary_emb=None, **kwargs):
        return self.processor(self, hidden_states, encoder_hidden_states,
                              attention_mask, rotary_emb, **kwargs)


class WanShiftWindow2DInferProcessor:
    """Mask-free 2D spatial shifted-window self-attention (full temporal view).

    Each window is densely pre-gathered (boundary-clamped) so attention reduces
    to a single SDPA-style call with no mask, padding or cyclic shift. Alternate
    layers use a half-window shift; the reverse step uses a priority-coherent
    scatter. RoPE is applied globally before partitioning.
    """

    def __init__(self, window_hw=(16, 16), shift_every_other_layer=True):
        wh, ww = window_hw
        assert wh > 0 and ww > 0
        self.window_hw = window_hw
        self.shift_every_other_layer = shift_every_other_layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is not None or getattr(attn, "is_cross_attention", False):
            raise RuntimeError("WanShiftWindow2DInferProcessor only supports self-attention.")
        if attention_mask is not None:
            raise RuntimeError("External attention_mask is not supported.")
        if not hasattr(attn, "_thw") or attn._thw is None:
            raise RuntimeError("attn._thw=(T,H,W) must be set before forward.")

        Tg, Hg, Wg = attn._thw
        B, K, _ = hidden_states.shape
        T, H, W = _infer_local_thw((Tg, Hg, Wg), K)
        if K != T * H * W:
            raise RuntimeError(f"K mismatch: K={K}, inferred T*H*W={T * H * W}.")

        cfg_wh, cfg_ww = self.window_hw
        wh, ww = min(cfg_wh, H), min(cfg_ww, W)

        do_shift = False
        if hasattr(attn, "_do_shift"):
            do_shift = bool(attn._do_shift)
        elif self.shift_every_other_layer and hasattr(attn, "_layer_id"):
            do_shift = (int(attn._layer_id) % 2 == 1)
        prefer_front = not do_shift

        meta = _WindowRuntimeMetaCache.get(
            T, H, W, wh, ww, do_shift=do_shift, prefer_front=prefer_front,
            device=hidden_states.device)
        Nw, Lw = meta.Nw, meta.Lw
        Hn = attn.heads
        Dh = attn.inner_dim // Hn

        query, key, value = _get_qkv_projections(attn, hidden_states, None)
        _release_input_storage(hidden_states)
        del hidden_states

        query = attn.norm_q(query).unflatten(2, (Hn, Dh))
        key = attn.norm_k(key).unflatten(2, (Hn, Dh))
        value = value.unflatten(2, (Hn, Dh))

        value = torch.index_select(value, 1, meta.lin_flat).view(B * Nw, Lw, Hn, Dh)

        if rotary_emb is not None:
            query = _apply_rotary_emb_inplace(query, *rotary_emb)
            key = _apply_rotary_emb_inplace(key, *rotary_emb)

        query = torch.index_select(query, 1, meta.lin_flat).view(B * Nw, Lw, Hn, Dh)
        key = torch.index_select(key, 1, meta.lin_flat).view(B * Nw, Lw, Hn, Dh)

        o_win = _dense_attn(query, key, value)
        del query, key, value

        o_flat = o_win.reshape(B, Nw * Lw, Hn, Dh)
        del o_win
        out = torch.index_select(o_flat, 1, meta.owner_pos)
        del o_flat

        out = out.reshape(B, K, Hn * Dh)
        out = attn.to_out[0](out)
        if attn.training and isinstance(attn.to_out[1], nn.Dropout) and attn.to_out[1].p > 0.0:
            out = attn.to_out[1](out)
        return out


# --------------------------------------------------------------------------- #
# Embeddings and transformer block                                            #
# --------------------------------------------------------------------------- #

class WanImageEmbedding(nn.Module):
    def __init__(self, in_features, out_features, pos_embed_seq_len=None):
        super().__init__()
        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        self.pos_embed = (nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
                          if pos_embed_seq_len is not None else None)

    def forward(self, encoder_hidden_states_image):
        if self.pos_embed is not None:
            B, S, C = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * S, C)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed
        x = self.norm1(encoder_hidden_states_image)
        x = self.ff(x)
        return self.norm2(x)


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(self, dim, time_freq_dim, time_proj_dim, text_embed_dim,
                 image_embed_dim=None, pos_embed_seq_len=None):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.image_embedder = (WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)
                               if image_embed_dim is not None else None)

    def forward(self, timestep, encoder_hidden_states,
                encoder_hidden_states_image=None, timestep_seq_len=None):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != dtype and dtype != torch.int8:
            timestep = timestep.to(dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)
        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(self, attention_head_dim, patch_size, max_seq_len, theta=10000.0):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self.t_dim, self.h_dim, self.w_dim = t_dim, h_dim, w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        freqs_cos, freqs_sin = [], []
        for dim in [t_dim, h_dim, w_dim]:
            fc, fs = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=True,
                repeat_interleave_real=True, freqs_dtype=freqs_dtype)
            freqs_cos.append(fc)
            freqs_sin.append(fs)
        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states):
        B, C, F_, H, W = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = F_ // p_t, H // p_h, W // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        fc_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        fc_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        fc_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        fs_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        fs_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        fs_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos_ = torch.cat([fc_f, fc_h, fc_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin_ = torch.cat([fs_f, fs_h, fs_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return freqs_cos_, freqs_sin_


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm="rms_norm_across_heads",
                 cross_attn_norm=False, eps=1e-6, added_kv_proj_dim=None):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
                                  cross_attention_dim_head=None, processor=WanAttnProcessor())
        self.attn2 = WanAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, eps=eps,
                                  added_kv_proj_dim=added_kv_proj_dim,
                                  cross_attention_dim_head=dim // num_heads, processor=WanAttnProcessor())
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        h_dtype = hidden_states.dtype

        if temb.ndim == 4:
            mods = (self.scale_shift_table.unsqueeze(0) + temb.float()).to(h_dtype)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mods.chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa = shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2)
            c_shift_msa, c_scale_msa, c_gate_msa = c_shift_msa.squeeze(2), c_scale_msa.squeeze(2), c_gate_msa.squeeze(2)
        else:
            mods = (self.scale_shift_table + temb.float()).to(h_dtype)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = mods.chunk(6, dim=1)

        attn_output = self.attn1(
            self.norm1(hidden_states).mul_(1.0 + scale_msa).add_(shift_msa), None, None, rotary_emb)
        hidden_states.addcmul_(attn_output, gate_msa)
        del attn_output

        attn_output = self.attn2(self.norm2(hidden_states), encoder_hidden_states, None, None)
        hidden_states.add_(attn_output)
        del attn_output

        ff_output = self.ffn(self.norm3(hidden_states).mul_(1.0 + c_scale_msa).add_(c_shift_msa))
        hidden_states.addcmul_(ff_output, c_gate_msa)
        del ff_output

        return hidden_states


# --------------------------------------------------------------------------- #
# Inference setup helpers                                                      #
# --------------------------------------------------------------------------- #

def enable_shifted_window_self_attention(model, window_hw=(16, 16)):
    """Fuse QKV projections and install the shifted-window self-attn processor."""
    proc = WanShiftWindow2DInferProcessor(window_hw=window_hw)
    for i, blk in enumerate(getattr(model, "blocks", [])):
        underlying = getattr(blk, "_orig_mod", blk)
        if hasattr(underlying, "attn1"):
            underlying.attn1._layer_id = i
    for _, m in model.named_modules():
        if isinstance(m, WanAttention):
            m.fuse_projections()
            if not getattr(m, "is_cross_attention", False):
                m.set_processor(proc)


def compile_transformer_blocks(model, mode="default"):
    if not hasattr(torch, "compile"):
        logger.warning("torch.compile not available (requires PyTorch 2.0+). Skipping.")
        return
    if mode == "reduce-overhead":
        logger.warning("compile_mode='reduce-overhead' is incompatible with the in-place "
                       "residuals; falling back to 'default'.")
        mode = "default"
    for i, blk in enumerate(getattr(model, "blocks", [])):
        if isinstance(blk, WanTransformerBlock):
            model.blocks[i] = torch.compile(blk, mode=mode, fullgraph=False)


# --------------------------------------------------------------------------- #
# Main model                                                                  #
# --------------------------------------------------------------------------- #

class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self, patch_size=(1, 2, 2), num_attention_heads=40, attention_head_dim=128,
                 in_channels=16, out_channels=16, text_dim=4096, freq_dim=256, ffn_dim=13824,
                 num_layers=40, cross_attn_norm=True, qk_norm="rms_norm_across_heads", eps=1e-6,
                 image_dim=None, added_kv_proj_dim=None, rope_max_seq_len=1024,
                 pos_embed_seq_len=None, enable_swa=True, self_attn_window_hw=(16, 16),
                 use_torch_compile=True, compile_mode="default"):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim, image_embed_dim=image_dim, pos_embed_seq_len=pos_embed_seq_len)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads, qk_norm,
                                cross_attn_norm, eps, added_kv_proj_dim)
            for _ in range(num_layers)])
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim ** 0.5)

        self.gradient_checkpointing = False
        self._enable_swa = enable_swa
        self._self_attn_window_hw = self_attn_window_hw

    def prepare_for_inference(self, attention_backend="auto", use_torch_compile=True, compile_mode="default"):
        backend = set_attention_backend(attention_backend)
        logger.info(f"Using attention backend: {backend} "
                    f"(available: {list_available_attention_backends()})")
        enable_shifted_window_self_attention(self, window_hw=self._self_attn_window_hw)
        if use_torch_compile:
            compile_transformer_blocks(self, mode=compile_mode)
        _WindowIndexCache.clear()
        _WindowRuntimeMetaCache.clear()
        self.eval()

    @torch.inference_mode()
    def forward(self, hidden_states, timestep, encoder_hidden_states,
                encoder_hidden_states_image=None, return_dict=True, attention_kwargs=None):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        B, C, F_, H, W = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        ppf, pph, ppw = F_ // p_t, H // p_h, W // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()

        ts_seq_len = None
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(timestep, encoder_hidden_states,
                                    encoder_hidden_states_image, timestep_seq_len=ts_seq_len))

        timestep_proj = (timestep_proj.unflatten(2, (6, -1)) if ts_seq_len is not None
                         else timestep_proj.unflatten(1, (6, -1)))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        thw_global = (ppf, pph, ppw)
        cfg_wh, cfg_ww = self._self_attn_window_hw
        dev = hidden_states.device
        _WindowRuntimeMetaCache.get(ppf, pph, ppw, min(cfg_wh, pph), min(cfg_ww, ppw),
                                    do_shift=False, prefer_front=True, device=dev)
        _WindowRuntimeMetaCache.get(ppf, pph, ppw, min(cfg_wh, pph), min(cfg_ww, ppw),
                                    do_shift=True, prefer_front=False, device=dev)

        for i, blk in enumerate(self.blocks):
            underlying = getattr(blk, "_orig_mod", blk)
            if hasattr(underlying, "attn1"):
                underlying.attn1._thw = thw_global
                underlying.attn1._layer_id = i

        for blk in self.blocks:
            hidden_states = blk(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        h_dtype = hidden_states.dtype
        if temb.ndim == 3:
            mods = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).to(h_dtype)
            shift, scale = mods.chunk(2, dim=2)
            shift, scale = shift.squeeze(2), scale.squeeze(2)
        else:
            mods = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).to(h_dtype)
            shift, scale = mods.chunk(2, dim=1)

        normed = self.norm_out(hidden_states)
        normed.mul_(1.0 + scale).add_(shift)
        hidden_states = self.proj_out(normed)
        del normed

        hidden_states = hidden_states.reshape(B, ppf, pph, ppw, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
