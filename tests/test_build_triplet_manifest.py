"""Tests for tools/build_triplet_manifest.py."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "build_triplet_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_triplet_manifest", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_TOOL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOL)


class BuildTripletManifestTest(unittest.TestCase):
    def _make_roots(self, root: Path):
        roots = {name: root / name for name in ("hr", "hq", "lr")}
        for path in roots.values():
            path.mkdir()
        return roots

    @staticmethod
    def _touch(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")

    def test_relative_stem_matches_nested_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for sample in ("scene_a/clip_001.mp4", "scene_b/clip_002.mkv"):
                for dataset_root in roots.values():
                    self._touch(dataset_root / sample)

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                val_fraction=0.0,
                strict=True,
            )

            self.assertEqual(
                [record.sample_id for record in records],
                ["scene_a/clip_001", "scene_b/clip_002"],
            )
            self.assertTrue(all(record.split == "train" for record in records))
            self.assertEqual(summary["triplet_count"], 2)
            self.assertEqual(summary["missing_counts"], {"hr": 0, "hq": 0, "lr": 0})

    def test_non_strict_keeps_only_complete_triplets_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for dataset_root in roots.values():
                self._touch(dataset_root / "complete.mp4")
            self._touch(roots["hr"] / "hr_only.mp4")

            records, summary = _TOOL.build_manifest(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                val_fraction=0.0,
                strict=False,
            )

            self.assertEqual([record.sample_id for record in records], ["complete"])
            self.assertEqual(summary["missing_counts"]["hr"], 0)
            self.assertEqual(summary["missing_counts"]["hq"], 1)
            self.assertEqual(summary["missing_counts"]["lr"], 1)

    def test_strict_rejects_missing_triplets(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            for dataset_root in roots.values():
                self._touch(dataset_root / "complete.mp4")
            self._touch(roots["lr"] / "lr_only.mp4")

            with self.assertRaisesRegex(ValueError, "identical triplet keys"):
                _TOOL.build_manifest(
                    hr_root=roots["hr"],
                    hq_root=roots["hq"],
                    lr_root=roots["lr"],
                    strict=True,
                )

    def test_basename_mode_rejects_duplicate_stems(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = self._make_roots(Path(tmp))
            self._touch(roots["hr"] / "a/clip.mp4")
            self._touch(roots["hr"] / "b/clip.mp4")
            self._touch(roots["hq"] / "clip.mp4")
            self._touch(roots["lr"] / "clip.mp4")

            with self.assertRaisesRegex(ValueError, "Duplicate match keys"):
                _TOOL.build_manifest(
                    hr_root=roots["hr"],
                    hq_root=roots["hq"],
                    lr_root=roots["lr"],
                    match_mode="basename_stem",
                )

    def test_split_is_deterministic_and_manifest_is_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roots = self._make_roots(root)
            for index in range(100):
                for dataset_root in roots.values():
                    self._touch(dataset_root / f"clip_{index:03d}.mp4")

            kwargs = dict(
                hr_root=roots["hr"],
                hq_root=roots["hq"],
                lr_root=roots["lr"],
                seed=17,
                val_fraction=0.2,
                test_fraction=0.1,
                strict=True,
            )
            first, first_summary = _TOOL.build_manifest(**kwargs)
            second, _ = _TOOL.build_manifest(**kwargs)
            self.assertEqual(first, second)
            self.assertEqual(sum(first_summary["split_counts"].values()), 100)

            output = root / "manifests/triplets.jsonl"
            _TOOL.write_manifest(first, first_summary, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertEqual(
                set(rows[0]),
                {"sample_id", "hr", "hq", "lr", "split"},
            )
            self.assertTrue(output.with_suffix(".jsonl.summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
