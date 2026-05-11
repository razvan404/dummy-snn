from __future__ import annotations

import os
import sys
from pathlib import Path

CSRC_DIR = Path(__file__).parent / "csrc"
_CPP_NAME = "spike_driven_conv.cpp"
_CUDA_NAME = "spike_driven_conv_cuda.cu"

BASE_CXX_FLAGS = ["-O3", "-std=c++17"]
NVCC_FLAGS = ["-O3", "--use_fast_math"]


def openmp_flags() -> tuple[list[str], list[str]]:
    if sys.platform == "darwin":
        for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
            if (Path(prefix) / "include" / "omp.h").exists():
                return (
                    ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                    [f"-L{prefix}/lib", "-lomp"],
                )
        return [], []
    if os.name == "posix":
        return ["-fopenmp"], ["-lgomp"]
    return [], []


def has_cuda_source() -> bool:
    return (CSRC_DIR / _CUDA_NAME).exists()


def detect_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and has_cuda_source()


def sources(*, use_cuda: bool, relative_to: Path | None = None) -> list[str]:
    paths = [CSRC_DIR / _CPP_NAME]
    if use_cuda and has_cuda_source():
        paths.append(CSRC_DIR / _CUDA_NAME)
    if relative_to is not None:
        return [str(p.relative_to(relative_to)) for p in paths]
    return [str(p) for p in paths]


def cxx_flags(*, use_cuda: bool) -> list[str]:
    flags = list(BASE_CXX_FLAGS) + openmp_flags()[0]
    if use_cuda and has_cuda_source():
        flags.append("-DWITH_CUDA")
    return flags


def link_flags() -> list[str]:
    return list(openmp_flags()[1])
