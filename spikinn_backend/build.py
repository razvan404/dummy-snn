from __future__ import annotations

import logging
from typing import Any

from torch.utils.cpp_extension import load

from ._build_flags import (
    NVCC_FLAGS, cxx_flags, detect_cuda, link_flags, sources,
)

logger = logging.getLogger(__name__)


def load_backend() -> Any | None:
    """Cached in ~/.cache/torch_extensions; returns None on build failure."""
    use_cuda = detect_cuda()
    try:
        return load(
            name="spikinn_backend_ext",
            sources=sources(use_cuda=use_cuda),
            extra_cflags=cxx_flags(use_cuda=use_cuda),
            extra_ldflags=link_flags(),
            extra_cuda_cflags=NVCC_FLAGS if use_cuda else None,
            with_cuda=use_cuda,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("spikinn_backend extension failed to build: %s", exc)
        return None
