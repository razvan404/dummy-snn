from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from applications.pipeline.cache import feature_cache_filename

LOGS_ROOT = Path("logs")

# Datasets whose log base differs from their CLI name.
_BASE = {"cifar10": "cifar10_whitened"}


def base_dir(dataset: str) -> str:
    return _BASE.get(dataset, dataset)


@dataclass(frozen=True)
class LayerSpec:
    num_filters: int
    t_obj: float
    kernel_size: int = 5
    backend: str = "gather"  # analytical-inference backend (gather is fastest on GPU)


@dataclass(frozen=True)
class RunSpec:
    """A run: (dataset, seed) plus an ordered stack of LayerSpec.

    Sole owner of where a run's artifacts live, so train/cache/refine/evaluate cannot
    disagree. The refinement target is always `layers[-1]`; earlier layers are the frozen
    prefix. Single-layer is the one-element case (`RunSpec.single`). t_obj strictly
    increases up the stack. Layout: layer-1 keeps the original `nf_/tobj_/seed_` path;
    deeper layers nest under it as `L{i}_nf{F}_tobj{t}` (so existing single-layer paths
    are unchanged and the fixed L1 is shared across an L2 sweep).
    """

    dataset: str
    seed: int
    layers: tuple[LayerSpec, ...] = field(default=())

    def __post_init__(self):
        if not self.layers:
            raise ValueError("RunSpec needs at least one LayerSpec")
        tobjs = [lyr.t_obj for lyr in self.layers]
        if any(b <= a for a, b in zip(tobjs, tobjs[1:])):
            raise ValueError(f"t_obj must strictly increase up the stack, got {tobjs}")

    @classmethod
    def single(cls, dataset: str, num_filters: int, t_obj: float, seed: int) -> "RunSpec":
        return cls(dataset, seed, (LayerSpec(num_filters, t_obj),))

    @property
    def target(self) -> LayerSpec:
        return self.layers[-1]

    def _stack_suffix(self, base: Path) -> Path:
        l1 = self.layers[0]
        p = base / f"nf_{l1.num_filters}" / f"tobj_{l1.t_obj:.2f}" / f"seed_{self.seed}"
        for i, lyr in enumerate(self.layers[1:], start=2):
            p = p / f"L{i}_nf{lyr.num_filters}_tobj{lyr.t_obj:.2f}"
        return p

    @property
    def model_dir(self) -> Path:
        return self._stack_suffix(LOGS_ROOT / base_dir(self.dataset) / "sweep")

    @property
    def prefix_dir(self) -> Path | None:
        """Dir of the frozen prefix model (the run minus its last layer). None if single-layer."""
        if len(self.layers) == 1:
            return None
        return RunSpec(self.dataset, self.seed, self.layers[:-1]).model_dir

    def cache_path(self, step_size: float, max_drift: float) -> Path:
        return self.model_dir / feature_cache_filename(step_size, max_drift)

    def refinement_dir(self, method: str, variant_tag: str) -> Path:
        d = LOGS_ROOT / "snn_weight_analysis" / method / variant_tag / self.dataset / f"seed_{self.seed}"
        for i, lyr in enumerate(self.layers[1:], start=2):
            d = d / f"L{i}_nf{lyr.num_filters}_tobj{lyr.t_obj:.2f}"
        return d
