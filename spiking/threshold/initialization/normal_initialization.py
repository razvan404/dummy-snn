import numpy as np

from .initialization import ThresholdInitialization


class NormalInitialization(ThresholdInitialization):
    def __init__(self, min_threshold: float, std_dev: float = 0.1):
        self.min_threshold = min_threshold
        self.std_dev = std_dev

    def initialize(self, threshold: float, shape: tuple[int] | int = 1):
        thresholds = np.random.normal(loc=threshold, scale=self.std_dev, size=shape)
        thresholds = np.maximum(thresholds, self.min_threshold)
        if shape == 1:
            return thresholds[0]
        return thresholds
