#!/usr/bin/env python3
"""Build a deterministic manifest for aligned HR/HQ/LR video triplets.

The expected dataset contains three roots with the same logical clips:

* HR: highest-quality high-resolution target, e.g. 3840x2160
* HQ: clean/downsampled same-resolution reference, e.g. 1280x720
* LR: degraded/compressed input, e.g. 1280x720

Files are matched by a configurable key and written as JSON Lines. No video is
decoded here; geometric/temporal alignment is checked separately by
``tools/audit_triplet_alignment.py``.

Example::

    python tools/build_triplet_manifest.py \
        --hr-root /data/vsr/HR_2160p \
        --hq-root /data/vsr/HQ_720p \
        --lr-root /data/vsr/LR_720p \
        --output manifests/vsr_triplets.jsonl \
        --val-fraction 0.05 \
        --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
)


@dataclass(frozen=True)
class TripletRecord:
    sample_id: str
    hr: str
    hq: str
    lr: str
    split: str


def parse_extensions(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = list(value)
    normalized = []
    for extension in values:
        extension = str(extension).strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.append(extension)
    if not normalized:
        raise ValueError("At least one media extension is required")
    return tuple(dict.fromkeys(normalized))


def sample_key(path: Path, root: Path, mode: str) -> str:
    if mode == "relative_stem":
        return path.relative_to(root).with_suffix("").as_posix()
    if mode == "basename_stem":
        return path.stem
    raise ValueError(f"Unsupported match mode: {mode}")


def index_media(
    root: Path,
    *,
    extensions: tuple[str, ...],
    match_mode: str,
) -> dict[str, Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")

    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        key = sample_key(path, root, match_mode)
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path.resolve()

    if duplicates:
        preview = {
            key: [str(path) for path in paths]
            for key, paths in list(sorted(duplicates.items()))[:10]
        }
        raise ValueError(
            f"Duplicate match keys under {root} using mode={match_mode}: "
            f"{json.dumps(preview, indent=2)}"
        )
    if not index:
        raise ValueError(
            f"No media files found under {root} with extensions={extensions}"
        )
    return index


def deterministic_split(
    sample_id: str,
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> str:
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def build_manifest(
    *,
    hr_root: Path,
    hq_root: Path,
    lr_root: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    match_mode: str = "relative_stem",
    seed: int = 0,
    val_fraction: float = 0.05,
    test_fraction: float = 0.0,
    strict: bool = False,
) -> tuple[list[TripletRecord], dict[str, object]]:
    extensions = parse_extensions(extensions)
    indices = {
        "hr": index_media(
            hr_root, extensions=extensions, match_mode=match_mode
        ),
        "hq": index_media(
            hq_root, extensions=extensions, match_mode=match_mode
        ),
        "lr": index_media(
            lr_root, extensions=extensions, match_mode=match_mode
        ),
    }

    all_keys = set().union(*(set(index) for index in indices.values()))
    common_keys = set.intersection(*(set(index) for index in indices.values()))
    missing = {
        name: sorted(all_keys - set(index))
        for name, index in indices.items()
    }
    if strict and any(missing.values()):
        counts = {name: len(keys) for name, keys in missing.items()}
        raise ValueError(
            "Dataset roots do not contain identical triplet keys: "
            f"missing_counts={counts}"
        )

    records = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    for key in sorted(common_keys):
        split = deterministic_split(
            key,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        split_counts[split] += 1
        records.append(
            TripletRecord(
                sample_id=key,
                hr=str(indices["hr"][key]),
                hq=str(indices["hq"][key]),
                lr=str(indices["lr"][key]),
                split=split,
            )
        )

    summary: dict[str, object] = {
        "hr_root": str(hr_root.resolve()),
        "hq_root": str(hq_root.resolve()),
        "lr_root": str(lr_root.resolve()),
        "match_mode": match_mode,
        "extensions": list(extensions),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "indexed_counts": {
            name: len(index) for name, index in indices.items()
        },
        "triplet_count": len(records),
        "split_counts": split_counts,
        "missing_counts": {
            name: len(keys) for name, keys in missing.items()
        },
        "missing_examples": {
            name: keys[:20] for name, keys in missing.items()
        },
    }
    return records, summary


def write_manifest(
    records: list[TripletRecord],
    summary: dict[str, object],
    output: Path,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-root", type=Path, required=True)
    parser.add_argument("--hq-root", type=Path, required=True)
    parser.add_argument("--lr-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--match-mode",
        choices=("relative_stem", "basename_stem"),
        default="relative_stem",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated media extensions",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any logical clip is missing from one of the three roots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, summary = build_manifest(
        hr_root=args.hr_root,
        hq_root=args.hq_root,
        lr_root=args.lr_root,
        extensions=parse_extensions(args.extensions),
        match_mode=args.match_mode,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        strict=args.strict,
    )
    if not records:
        raise RuntimeError("No complete HR/HQ/LR triplets were found")
    write_manifest(records, summary, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
