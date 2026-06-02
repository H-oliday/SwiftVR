"""Fixed-size causal chunk protocol for streaming restoration.

A clip of length ``t = 4a + 1`` is split into one FIRST chunk, zero or more
MIDDLE chunks and one LAST chunk so that the total number of input frames equals
the total number of output frames. ``clip_len`` is the MIDDLE chunk size and
must be a multiple of 4.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List


class ChunkType(Enum):
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


@dataclass
class ChunkSpec:
    ctype: ChunkType
    frame_start: int
    frame_count: int
    b: int                 # LAST only: 4b + 1 input frames -> b + 1 latents
    clip_idx: int
    is_first_decode: bool  # trim the decoder's causal-padding head frames


def build_chunk_specs(t: int, clip_len: int) -> List[ChunkSpec]:
    assert clip_len % 4 == 0, f"clip_len must be a multiple of 4, got {clip_len}"

    if t <= clip_len + 4:
        return [ChunkSpec(ChunkType.LAST, 0, t, (t - 1) // 4, 0, True)]

    specs = [ChunkSpec(ChunkType.FIRST, 0, clip_len + 4, 0, 0, True)]

    remaining = t - (clip_len + 4)
    pos = clip_len + 4
    cidx = 1
    while remaining > 0:
        if remaining <= clip_len:
            specs.append(ChunkSpec(ChunkType.LAST, pos, remaining, (remaining - 1) // 4, cidx, False))
            break
        specs.append(ChunkSpec(ChunkType.MIDDLE, pos, clip_len, 0, cidx, False))
        remaining -= clip_len
        pos += clip_len
        cidx += 1

    return specs
