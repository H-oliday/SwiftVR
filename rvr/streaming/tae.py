"""Streaming wrapper around the Restoration-aware Autoencoder.

It runs the encoder/decoder clip-by-clip while passing the MemBlock and TPool
boundary state across chunks, so the result is identical to encoding/decoding
the whole clip at once.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from ..models.reae import MemBlock, TPool, TGrow
from .chunk import ChunkSpec, ChunkType


def apply_parallel_with_boundary(model, x, state=None):
    """Run ``model`` (a Sequential of streaming blocks) over ``x``.

    ``x`` has shape ``[N, T, C, H, W]``. ``state`` carries the MemBlock/TPool
    boundary buffers from the previous chunk; the updated state is returned.
    """
    if state is None:
        state = {}
    new_state = {}
    N, T, C, H, W = x.shape
    x = x.reshape(N * T, C, H, W)

    for i, b in enumerate(model):
        if isinstance(b, MemBlock):
            NT, C, H, W = x.shape
            T_ = NT // N
            _x = x.reshape(N, T_, C, H, W)
            key = f"mem_{i}"
            if key in state:
                mem = torch.cat([state[key], _x[:, :-1]], dim=1)
            else:
                mem = F.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T_]
            new_state[key] = _x[:, -1:].detach().clone()
            x = b(x, mem.reshape(NT, C, H, W))

        elif isinstance(b, TPool):
            NT, C, H, W = x.shape
            T_ = NT // N
            _x = x.reshape(N, T_, C, H, W)
            key = f"tpool_{i}"
            if key in state and state[key] is not None:
                _x = torch.cat([state[key], _x], dim=1)
                T_ = _x.shape[1]
            n_full = (T_ // b.stride) * b.stride
            rem = T_ - n_full
            new_state[key] = _x[:, n_full:].detach().clone() if rem > 0 else None
            if n_full > 0:
                x = b(_x[:, :n_full].reshape(N * n_full, C, H, W))
            else:
                return None, new_state

        elif isinstance(b, TGrow):
            x = b(x)
        else:
            x = b(x)

    NT, C, H, W = x.shape
    return x.view(N, NT // N, C, H, W), new_state


class StreamingTAE:
    def __init__(self, model):
        self.model = model
        self._enc_st = None
        self._dec_st = None
        self._enc_left = None
        self._first_dec = True

    def reset(self):
        self._enc_st = self._dec_st = None
        self._enc_left = None
        self._first_dec = True

    # ----- Fixed-size chunk interface (offline, frame-count preserving) ----- #

    @torch.no_grad()
    def encode_chunk_fixed(self, x: torch.Tensor, spec: ChunkSpec) -> torch.Tensor:
        ps = self.model.patch_size
        if ps > 1:
            N, T, C, H, W = x.shape
            x = F.pixel_unshuffle(x.reshape(N * T, C, H, W), ps)
            x = x.reshape(N, T, *x.shape[1:])

        if spec.ctype == ChunkType.LAST:
            x = torch.cat([x, x[:, -1:].expand(-1, 3, -1, -1, -1)], dim=1)

        z, self._enc_st = apply_parallel_with_boundary(self.model.encoder, x, self._enc_st)
        return z

    @torch.no_grad()
    def decode_chunk_fixed(self, z: torch.Tensor, spec: ChunkSpec) -> Optional[torch.Tensor]:
        x, self._dec_st = apply_parallel_with_boundary(self.model.decoder, z, self._dec_st)
        if x is None:
            return None
        x = self._postprocess(x)
        if spec.is_first_decode:
            x = x[:, self.model.frames_to_trim:]
        return x

    # ----- Generic streaming interface (online, arbitrary chunk lengths) ---- #

    @torch.no_grad()
    def encode_chunk(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        ps = self.model.patch_size
        if ps > 1:
            N, T, C, H, W = x.shape
            x = F.pixel_unshuffle(x.reshape(N * T, C, H, W), ps)
            x = x.reshape(N, T, *x.shape[1:])
        if self._enc_left is not None:
            x = torch.cat([self._enc_left, x], dim=1)
            self._enc_left = None
        T = x.shape[1]
        rem = T % 4
        if rem:
            keep = T - rem
            if keep > 0:
                self._enc_left = x[:, keep:].detach().clone()
                x = x[:, :keep]
            else:
                self._enc_left = x.detach().clone()
                return None
        z, self._enc_st = apply_parallel_with_boundary(self.model.encoder, x, self._enc_st)
        return z

    @torch.no_grad()
    def flush_encoder(self) -> Optional[torch.Tensor]:
        if self._enc_left is None:
            return None
        x = self._enc_left
        self._enc_left = None
        T = x.shape[1]
        if T % 4:
            p = 4 - T % 4
            x = torch.cat([x, x[:, -1:].expand(-1, p, -1, -1, -1)], dim=1)
        z, self._enc_st = apply_parallel_with_boundary(self.model.encoder, x, self._enc_st)
        return z

    @torch.no_grad()
    def decode_chunk(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        x, self._dec_st = apply_parallel_with_boundary(self.model.decoder, z, self._dec_st)
        if x is None:
            return None
        x = self._postprocess(x)
        if self._first_dec:
            x = x[:, self.model.frames_to_trim:]
            self._first_dec = False
        return x

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0, 1)
        ps = self.model.patch_size
        if ps > 1:
            N, T, C, H, W = x.shape
            x = F.pixel_shuffle(x.reshape(N * T, C, H, W), ps)
            x = x.reshape(N, T, *x.shape[1:])
        return x
