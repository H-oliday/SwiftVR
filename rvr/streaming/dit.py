"""Streaming one-step DiT for RVR.

RVR collapses iterative diffusion sampling to a single forward pass taken at the
fully-degraded endpoint of the flow (t = 1). No sampling scheduler is therefore
needed at inference time: the conditioning timestep is the constant
``INFERENCE_TIMESTEP`` below, equivalent to a flow-matching schedule evaluated by
``set_timesteps(1)`` (whose ``scale_model_input`` is the identity). Adjust it if
your training schedule uses a different number of train timesteps.
"""

from typing import Optional

import torch

from .chunk import ChunkSpec

INFERENCE_TIMESTEP = 999.0
ROPE_EXTEND_MARGIN = 256


def _ensure_rope_cache_len(rope, required_len):
    cos, sin = rope.freqs_cos, rope.freqs_sin
    old_len = cos.shape[0]
    if required_len <= old_len:
        return
    if old_len < 2:
        raise RuntimeError(f"Cannot extend RoPE cache: length={old_len}")
    with torch.no_grad():
        dev, old_dtype = cos.device, cos.dtype
        cos64, sin64 = cos.to(torch.float64), sin.to(torch.float64)
        cos0, sin0, cos1, sin1 = cos64[0:1], sin64[0:1], cos64[1:2], sin64[1:2]
        cos_delta = cos1 * cos0 + sin1 * sin0
        sin_delta = sin1 * cos0 - cos1 * sin0
        angle0 = torch.atan2(sin0, cos0)
        step = torch.atan2(sin_delta, cos_delta)
        pos = torch.arange(old_len, required_len, device=dev, dtype=torch.float64).view(-1, 1)
        angle = angle0 + pos * step
        rope.freqs_cos = torch.cat([cos, torch.cos(angle).to(old_dtype)], 0).contiguous()
        rope.freqs_sin = torch.cat([sin, torch.sin(angle).to(old_dtype)], 0).contiguous()


def _rope_with_offset(rope, ppf, pph, ppw, t_off=0, h_off=0, w_off=0):
    required_len = max(t_off + ppf, h_off + pph, w_off + ppw)
    if required_len > rope.freqs_cos.shape[0]:
        required_len += max(0, int(ROPE_EXTEND_MARGIN))
    _ensure_rope_cache_len(rope, required_len)
    sp = [rope.t_dim, rope.h_dim, rope.w_dim]
    fc = rope.freqs_cos.split(sp, dim=1)
    fs = rope.freqs_sin.split(sp, dim=1)
    cf = fc[0][t_off:t_off + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    ch = fc[1][h_off:h_off + pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    cw = fc[2][w_off:w_off + ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    sf = fs[0][t_off:t_off + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    sh = fs[1][h_off:h_off + pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    sw = fs[2][w_off:w_off + ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
    return (torch.cat([cf, ch, cw], -1).reshape(1, ppf * pph * ppw, 1, -1),
            torch.cat([sf, sh, sw], -1).reshape(1, ppf * pph * ppw, 1, -1))


def _precompute_cond(transformer, B, prompt_emb, timestep):
    pe = prompt_emb.clone()
    if pe.ndim == 2:
        pe = pe.unsqueeze(0).expand(B, -1, -1)
    elif pe.shape[0] != B:
        pe = pe.expand(B, -1, -1)
    ts = timestep.clone()
    ts_seq = None
    if ts.ndim == 2:
        ts_seq = ts.shape[1]
        ts = ts.flatten()
    temb, tp, enc_hs, enc_img = transformer.condition_embedder(ts, pe, None, timestep_seq_len=ts_seq)
    tp = tp.unflatten(2 if ts_seq else 1, (6, -1))
    if enc_img is not None:
        enc_hs = torch.cat([enc_img, enc_hs], dim=1)
    return temb, tp, enc_hs


@torch.inference_mode()
def _dit_forward_chunk(transformer, chunk, temb, tp, enc_hs, t_off=0):
    """One forward pass of the DiT, returning the predicted degradation velocity."""
    p_t, p_h, p_w = transformer.config.patch_size
    B, C, F, H, W = chunk.shape
    ppf, pph, ppw = F // p_t, H // p_h, W // p_w
    rope = _rope_with_offset(transformer.rope, ppf, pph, ppw, t_off)
    hs = transformer.patch_embedding(chunk).flatten(2).transpose(1, 2)

    thw_global = (ppf, pph, ppw)
    for i, blk in enumerate(transformer.blocks):
        underlying = getattr(blk, "_orig_mod", blk)
        if hasattr(underlying, "attn1"):
            underlying.attn1._thw = thw_global
            underlying.attn1._layer_id = i
    for blk in transformer.blocks:
        hs = blk(hs, enc_hs, tp, rope)

    if temb.ndim == 3:
        shift, scale = (transformer.scale_shift_table.unsqueeze(0).to(temb.device)
                        + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
    else:
        shift, scale = (transformer.scale_shift_table.to(temb.device)
                        + temb.unsqueeze(1)).chunk(2, dim=1)
    hs = (transformer.norm_out(hs.float()) * (1 + scale.to(hs.device)) + shift.to(hs.device)).type_as(hs)
    hs = transformer.proj_out(hs)
    hs = hs.reshape(B, ppf, pph, ppw, p_t, p_h, p_w, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hs.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class StreamingDiT:
    """One-step DiT with temporal overlap blending across chunks."""

    def __init__(self, transformer, overlap=0):
        self.transformer = transformer
        self.overlap = overlap
        self._prev_lq = self._prev_out = None
        self._g_off = 0
        self._cond_cache_key = self._cond_cache = None

    def reset(self):
        self._prev_lq = self._prev_out = None
        self._g_off = 0

    def _get_cached_condition(self, B, prompt_emb, dev, dt):
        cache_key = (int(B), dev.type, dev.index, str(dt), tuple(prompt_emb.shape))
        if self._cond_cache_key != cache_key or self._cond_cache is None:
            ts = torch.full((B,), INFERENCE_TIMESTEP, device=dev, dtype=torch.float32)
            self._cond_cache = _precompute_cond(self.transformer, B, prompt_emb.to(dev, dt), ts)
            self._cond_cache_key = cache_key
        return self._cond_cache

    @torch.inference_mode()
    def denoise(self, lq, prompt_emb):
        """Restore one chunk of latents ``lq`` (shape [B, C, F, H, W])."""
        dev, dt = lq.device, lq.dtype
        B, C, F_cur, H, W = lq.shape

        ol = 0
        if self._prev_lq is not None and self.overlap > 0:
            ol = self._prev_lq.shape[2]
            lq_ext = torch.cat([self._prev_lq.to(dev), lq], dim=2)
            t_rope = self._g_off - ol
        else:
            lq_ext = lq
            t_rope = self._g_off

        temb, tp, enc_hs = self._get_cached_condition(B, prompt_emb, dev, dt)
        pred = _dit_forward_chunk(self.transformer, lq_ext, temb, tp, enc_hs, t_off=t_rope)
        den_ext = lq_ext - pred

        if ol > 0 and self._prev_out is not None:
            ramp = torch.linspace(0, 1, ol, device=dev, dtype=dt).view(1, 1, ol, 1, 1)
            den_ext[:, :, :ol] = self._prev_out.to(dev) * (1 - ramp) + den_ext[:, :, :ol] * ramp
            den_out = den_ext[:, :, ol:]
        else:
            den_out = den_ext

        k = min(self.overlap, F_cur)
        if k > 0:
            self._prev_lq = lq[:, :, -k:].detach().cpu().clone()
            self._prev_out = den_out[:, :, -k:].detach().cpu().clone()
        else:
            self._prev_lq = self._prev_out = None

        self._g_off += F_cur
        return den_out

    @torch.inference_mode()
    def denoise_last_chunk(self, z_new_ntchw, spec: ChunkSpec, prompt_emb,
                           prev_dit_out_cpu: Optional[torch.Tensor], n_lat: int, device, dtype):
        """LAST chunk: pad the (b+1) new latents up to (n_lat+1) with the previous
        chunk's latents (or zeros) for a correct RoPE offset, run one pass, and
        keep only the new ``b+1`` denoised latents.
        """
        lat_count = spec.b + 1
        pad_count = (n_lat + 1) - lat_count

        z_bcfhw = z_new_ntchw.permute(0, 2, 1, 3, 4).contiguous()
        if pad_count > 0:
            if prev_dit_out_cpu is not None:
                pad_z = prev_dit_out_cpu[:, :, -pad_count:].to(device=device, dtype=dtype)
            else:
                pad_z = torch.zeros(z_bcfhw.shape[0], z_bcfhw.shape[1], pad_count,
                                    z_bcfhw.shape[3], z_bcfhw.shape[4], device=device, dtype=dtype)
            z_bcfhw = torch.cat([pad_z, z_bcfhw], dim=2)

        t_off = max(0, self._g_off - pad_count)
        temb, tp, enc_hs = self._get_cached_condition(z_bcfhw.shape[0], prompt_emb, device, dtype)
        pred = _dit_forward_chunk(self.transformer, z_bcfhw, temb, tp, enc_hs, t_off=t_off)
        z_den = (z_bcfhw - pred)[:, :, -lat_count:].contiguous()

        self._g_off += lat_count
        return z_den.permute(0, 2, 1, 3, 4).contiguous()
