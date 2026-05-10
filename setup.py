"""AOT build for the spike-driven C++/CUDA extension.

Compiled into ``spiking_backend._ext`` so ``import spiking_backend`` works
without a JIT step. Falls through to ``spiking_backend.build.load_backend``
(JIT) and finally to the pure-PyTorch reference if the compiled module
isn't importable.
"""

from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch

# Relative paths — setuptools rejects absolute paths in extension sources.
CSRC = Path("spiking_backend") / "csrc"


def _build_extension():
    sources = [str(CSRC / "spike_driven_conv.cpp")]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_ldflags: list[str] = []
    if os.name == "posix":
        extra_cflags.append("-fopenmp")
        extra_ldflags.append("-lgomp")

    use_cuda = torch.cuda.is_available()
    cuda_src = Path(__file__).parent / CSRC / "spike_driven_conv_cuda.cu"
    if use_cuda and cuda_src.exists():
        sources.append(str(CSRC / "spike_driven_conv_cuda.cu"))
        extra_cflags.append("-DWITH_CUDA")
        return CUDAExtension(
            name="spiking_backend._ext",
            sources=sources,
            extra_compile_args={
                "cxx": extra_cflags,
                "nvcc": ["-O3", "--use_fast_math"],
            },
            extra_link_args=extra_ldflags,
        )

    return CppExtension(
        name="spiking_backend._ext",
        sources=sources,
        extra_compile_args=extra_cflags,
        extra_link_args=extra_ldflags,
    )


setup(
    ext_modules=[_build_extension()],
    cmdclass={"build_ext": BuildExtension},
)
