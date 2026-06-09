"""Packaging for the SwiftVR inference package."""

from pathlib import Path
from setuptools import setup, find_packages

_here = Path(__file__).parent
_long_description = (_here / "README.md").read_text(encoding="utf-8") if (_here / "README.md").exists() else ""


def _read_requirements():
    reqs = []
    for line in (_here / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="swiftvr",
    version="0.1.0",
    description="One-step generative streaming real-time video restoration.",
    long_description=_long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["swiftvr", "swiftvr.*"]),
    python_requires=">=3.10",
    install_requires=_read_requirements(),
    license="Apache-2.0",
)
