"""Restoration-aware Autoencoder (ReAE).

A lightweight causal-streaming encoder/decoder used by RVR to move between pixel
space and the DiT latent space. The memory-block topology is adapted from TAEHV
(https://github.com/madebyollin/taehv, MIT License).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual block that fuses the current frame with the previous one."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = (nn.Conv2d(n_in, n_out, 1, bias=False)
                     if n_in != n_out else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    """Temporal pooling by a factor of ``stride`` via a 1x1 channel mix."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    """Temporal upsampling by a factor of ``stride``.

    ``stride == 1`` is a plain 1x1 projection; ``stride == 2`` upsamples in time
    with nearest interpolation followed by a depth-only 3D convolution. ``conv``
    is kept solely for checkpoint compatibility and is not used at inference.
    """

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.n_f = n_f
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

        if stride == 1:
            self.proj = nn.Conv2d(n_f, n_f, 1, bias=False)
            self.conv3d = None
        else:
            self.conv3d = nn.Conv3d(n_f, n_f, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
            self.proj = None

    def forward(self, x):
        NT, C, H, W = x.shape
        if self.stride == 1:
            return self.proj(x)
        x = x.unsqueeze(2)
        x = F.interpolate(x, size=(self.stride, H, W), mode='nearest')
        x = self.conv3d(x)
        return x.permute(0, 2, 1, 3, 4).reshape(NT * self.stride, C, H, W)


class ReAE(nn.Module):
    def __init__(
        self,
        checkpoint_path=None,
        width_mult=2,
        decoder_time_upscale=(True, True),
        decoder_space_upscale=(True, True, True),
        patch_size=2,
        latent_channels=48,
    ):
        super().__init__()
        self.width_mult = width_mult
        self.image_channels = 3
        self.patch_size = patch_size
        self.latent_channels = latent_channels

        e_enc = 64

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size ** 2, e_enc),
            nn.ReLU(inplace=True),
            TPool(e_enc, 2),
            conv(e_enc, e_enc, stride=2, bias=False),
            MemBlock(e_enc, e_enc),
            MemBlock(e_enc, e_enc),
            MemBlock(e_enc, e_enc),
            TPool(e_enc, 2),
            conv(e_enc, e_enc, stride=2, bias=False),
            MemBlock(e_enc, e_enc),
            MemBlock(e_enc, e_enc),
            MemBlock(e_enc, e_enc),
            TPool(e_enc, 1),
            conv(e_enc, e_enc, stride=2, bias=False),
            MemBlock(e_enc, e_enc),
            MemBlock(e_enc, e_enc),
            MemBlock(e_enc, e_enc),
            conv(e_enc, self.latent_channels),
        )

        n_f = [256 * width_mult, 128 * width_mult, 64 * width_mult, 64]
        self.frames_to_trim = 2 ** sum(decoder_time_upscale) - 1

        self.decoder = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),

            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
            TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
            conv(n_f[1], n_f[2], bias=False),

            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
            TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
            conv(n_f[2], n_f[3], bias=False),

            nn.ReLU(inplace=True),
            conv(n_f[3], self.image_channels * self.patch_size ** 2),
        )

        if checkpoint_path is not None:
            self.load_state_dict(load_file(checkpoint_path, device="cpu"), strict=True)
