"""AOT build for the spike-driven C++/CUDA extension.

Compiled into ``spiking_backend._ext`` so ``import spiking_backend`` works
without a JIT step. Falls through to ``spiking_backend.build.load_backend``
(JIT) and finally to the pure-PyTorch reference if the compiled module
isn't importable. Compiler flags come from ``spiking_backend._build_flags``,
which is also used by the JIT path so the two binaries can't drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

_HERE = Path(__file__).parent

# Direct file load — going through ``spiking_backend.__init__`` would trigger
# the JIT build path while we're still setting up the AOT build.
_BF_SPEC = importlib.util.spec_from_file_location(
    "_build_flags", _HERE / "spiking_backend" / "_build_flags.py",
)
_bf = importlib.util.module_from_spec(_BF_SPEC)
_BF_SPEC.loader.exec_module(_bf)


def _build_extension():
    use_cuda = _bf.detect_cuda()
    srcs = _bf.sources(use_cuda=use_cuda, relative_to=_HERE)
    cxx = _bf.cxx_flags(use_cuda=use_cuda)
    ld = _bf.link_flags()
    if use_cuda:
        return CUDAExtension(
            name="spiking_backend._ext",
            sources=srcs,
            extra_compile_args={"cxx": cxx, "nvcc": _bf.NVCC_FLAGS},
            extra_link_args=ld,
        )
    return CppExtension(
        name="spiking_backend._ext",
        sources=srcs,
        extra_compile_args=cxx,
        extra_link_args=ld,
    )


setup(
    ext_modules=[_build_extension()],
    cmdclass={"build_ext": BuildExtension},
)
