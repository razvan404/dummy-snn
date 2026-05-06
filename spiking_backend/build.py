"""JIT loader for the spike-driven conv accumulator C++/CUDA module.

First import compiles the C++/CUDA sources under
``~/.cache/torch_extensions/spiking_backend/``. Subsequent imports reuse
the cached shared library.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

_CSRC_DIR = Path(__file__).parent / "csrc"


def load_backend() -> Any | None:
    """Compile and load the spike-driven-conv extension; ``None`` on failure."""
    sources = [str(_CSRC_DIR / "spike_driven_conv.cpp")]

    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags: list[str] = []
    # OpenMP for batched parallelism. Linux toolchains use -fopenmp / libgomp.
    if os.name == "posix":
        extra_cflags.append("-fopenmp")
        extra_ldflags.append("-lgomp")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cuda_src = _CSRC_DIR / "spike_driven_conv.cu"
        if cuda_src.exists():
            sources.append(str(cuda_src))

    try:
        return load(
            name="spiking_backend_ext",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_cuda_cflags=["-O3"] if use_cuda else None,
            with_cuda=use_cuda and (_CSRC_DIR / "spike_driven_conv.cu").exists(),
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - build env issues
        logger.warning("spiking_backend extension failed to build: %s", exc)
        return None
