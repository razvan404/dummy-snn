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
    """JIT-compile the extension; ``None`` on build failure (cached under ``~/.cache/torch_extensions``)."""
    sources = [str(_CSRC_DIR / "spike_driven_conv.cpp")]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags: list[str] = []
    if os.name == "posix":
        extra_cflags.append("-fopenmp")
        extra_ldflags.append("-lgomp")

    use_cuda = torch.cuda.is_available()
    has_cuda_src = (_CSRC_DIR / "spike_driven_conv.cu").exists()
    if use_cuda and has_cuda_src:
        sources.append(str(_CSRC_DIR / "spike_driven_conv.cu"))
        extra_cflags.append("-DWITH_CUDA")

    try:
        return load(
            name="spiking_backend_ext",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_cuda_cflags=["-O3", "--use_fast_math"] if use_cuda else None,
            with_cuda=use_cuda and has_cuda_src,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("spiking_backend extension failed to build: %s", exc)
        return None
