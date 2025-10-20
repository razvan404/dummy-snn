from abc import ABC, abstractmethod

import torch


class CallbacksInterface(ABC):
    @abstractmethod
    def callback_step_spike(
        self,
        batch_idx: int,
        current_time: float,
        output_spike: torch.Tensor,
        split: str,
    ) -> bool:
        """
        Returns if further spikes should be logged.
        """
        pass

    @abstractmethod
    def callback_step(
        self,
        batch_idx: int,
        pre_spike_times: torch.Tensor,
        dw: float,
        label: str | None,
        split: str,
    ):
        pass
