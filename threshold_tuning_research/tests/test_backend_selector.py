import pytest
import torch

from threshold_tuning_research.backend_selector import select_backend


@pytest.mark.parametrize("device", ["cuda", torch.device("cuda:0")])
@pytest.mark.parametrize("first_spike_only", [True, False])
def test_cuda_always_picks_gather(device, first_spike_only) -> None:
    for B in (1, 8, 32, 128, 1024):
        assert select_backend(device, B, first_spike_only=first_spike_only) == "gather"


@pytest.mark.parametrize("first_spike_only", [True, False])
def test_cpu_always_picks_gather(first_spike_only: bool) -> None:
    for B in (1, 4, 7, 8, 128, 1024):
        assert select_backend("cpu", B, first_spike_only) == "gather"
