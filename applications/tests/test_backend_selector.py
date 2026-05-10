import pytest
import torch

from applications.backend_selector import select_backend


@pytest.mark.parametrize("device", ["cuda", torch.device("cuda:0")])
@pytest.mark.parametrize("first_spike_only", [True, False])
def test_cuda_always_picks_gather(device, first_spike_only) -> None:
    for B in (1, 8, 32, 128, 1024):
        assert select_backend(device, B, first_spike_only=first_spike_only) == "gather"


@pytest.mark.parametrize("first_spike_only", [True, False])
def test_cpu_threshold(first_spike_only: bool) -> None:
    assert select_backend("cpu", 1, first_spike_only) == "gather"
    assert select_backend("cpu", 4, first_spike_only) == "gather"
    assert select_backend("cpu", 7, first_spike_only) == "gather"
    assert select_backend("cpu", 8, first_spike_only) == "scatter"
    assert select_backend("cpu", 128, first_spike_only) == "scatter"
    assert select_backend("cpu", 1024, first_spike_only) == "scatter"
