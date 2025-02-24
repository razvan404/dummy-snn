from .adaptation import ThresholdAdaptation


class FalezAdaptation(ThresholdAdaptation):
    def __init__(
        self,
        min_threshold: float,
        threshold_learning_rate: float,
        target_timestamp: float,
    ):
        super().__init__()
        self.min_threshold = min_threshold
        self.threshold_learning_rate = threshold_learning_rate
        self.target_timestamp = target_timestamp

    def update(self, current_threshold: float, current_time: float) -> float:
        return max(
            self.min_threshold,
            current_threshold
            - self.threshold_learning_rate * (current_time - self.target_timestamp),
        )
