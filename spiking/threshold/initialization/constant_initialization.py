import numpy as np

from .initialization import ThresholdInitialization


class ConstantInitialization(ThresholdInitialization):
    def initialize(self, threshold: float, shape: tuple[int] | int = 1):
        if shape == 1:
            return threshold
        return np.ones(shape) * threshold
