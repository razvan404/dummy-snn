import numpy as np
import unittest

from ..learning import STDP
from ..competition import WinnerTakesAll
from ..layers import IntegrateAndFireLayer, IntegrateAndFireOptimizedLayer
from ..threshold import (
    NormalInitialization,
    FalezAdaptation,
    CompetitiveFalezAdaptation,
)


class TestOptimizedIntegrateAndFireLayer(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_outputs = 5
        self.learning_mechanism = STDP()
        self.competition_mechanism = WinnerTakesAll()
        self.threshold = 0.8
        self.refractory_period = np.inf
        self.min_threshold = 0.2
        self.threshold_initialization = NormalInitialization(self.min_threshold)
        self.target_timestamp = 0.7
        self.falez_adaptation = FalezAdaptation(
            self.min_threshold,
            learning_rate=2e-2,
            target_timestamp=self.target_timestamp,
        )
        self.competitive_falez_adaptation = CompetitiveFalezAdaptation(
            self.min_threshold, learning_rate=2e-2
        )

        self.layer = IntegrateAndFireLayer(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            learning_mechanism=self.learning_mechanism,
            competition_mechanism=self.competition_mechanism,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            threshold_adaptation=self.competitive_falez_adaptation,
            threshold_initialization=self.threshold_initialization,
        )

        self.optimized_layer = IntegrateAndFireOptimizedLayer(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            learning_mechanism=self.learning_mechanism,
            competition_mechanism=self.competition_mechanism,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            threshold_adaptation=self.competitive_falez_adaptation,
            threshold_initialization=self.threshold_initialization,
        )

        self.optimized_layer.weights = np.array(
            [neuron.weights for neuron in self.layer.neurons]
        )
        self.optimized_layer.thresholds = np.array(
            [neuron.threshold for neuron in self.layer.neurons]
        )
        self.max_iter = 100

    def test_layer_equivalence(self):
        incoming_spikes = np.random.randint(0, 2, size=(self.num_inputs,)).astype(
            np.float32
        )
        dt = 0.05
        target_spike_times = np.random.uniform(
            self.target_timestamp - 0.2,
            self.target_timestamp + 0.2,
            size=(self.num_inputs,),
        )
        target_spike_times[incoming_spikes == 0.0] = np.inf

        print(
            [neuron.threshold for neuron in self.layer.neurons],
            self.optimized_layer.thresholds,
        )
        print("Incoming spikes:", incoming_spikes)
        print("Target spike times:", target_spike_times)

        for iteration in range(20):
            print(f"Iteration {iteration}")
            np.testing.assert_array_almost_equal(
                [neuron.weights for neuron in self.layer.neurons],
                self.optimized_layer.weights,
                err_msg=f"Weights do not match between layers (iter={iteration}).",
            )

            np.testing.assert_array_almost_equal(
                [neuron.threshold for neuron in self.layer.neurons],
                self.optimized_layer.thresholds,
                err_msg=f"Threshold do not match between layers (iter={iteration}).",
            )

            num_iter_layer = 0
            current_time_layer = 0
            while (
                not np.any(
                    spikes_layer := self.layer.forward(
                        incoming_spikes, current_time_layer, dt
                    )
                )
                and current_time_layer < self.max_iter
            ):
                num_iter_layer += 1
                current_time_layer = current_time_layer + dt

            num_iter_optimized = 0
            current_time_optimized = 0
            while (
                not np.any(
                    spikes_optimized := self.optimized_layer.forward(
                        incoming_spikes, current_time_optimized, dt
                    )
                )
                and current_time_layer < self.max_iter
            ):
                num_iter_optimized += 1
                current_time_optimized = current_time_optimized + dt

            np.testing.assert_equal(
                num_iter_layer,
                num_iter_optimized,
                err_msg=f"Output spikes iteration count do not match between layers (iter={iteration}).",
            )

            np.testing.assert_array_almost_equal(
                self.layer.spike_times,
                self.optimized_layer.spike_times,
                err_msg=f"Spike times do not match between layers (iter={iteration})",
            )

            print(
                "Spike times:", self.layer.spike_times, self.optimized_layer.spike_times
            )

            np.testing.assert_array_almost_equal(
                spikes_layer,
                spikes_optimized,
                err_msg=f"Output Spike times do not match between layers (iter={iteration})",
            )

            print("Output spikes:", spikes_layer, spikes_optimized)

            np.testing.assert_array_almost_equal(
                [neuron.membrane_potential for neuron in self.layer.neurons],
                self.optimized_layer.membrane_potentials,
                err_msg=f"Membrane potentials do not match between layers (iter={iteration})",
            )

            self.layer.backward(target_spike_times)
            self.optimized_layer.backward(target_spike_times)

            print(
                "Thresholds",
                np.array([neuron.threshold for neuron in self.layer.neurons]),
                self.optimized_layer.thresholds,
            )

            np.testing.assert_array_almost_equal(
                [neuron.weights for neuron in self.layer.neurons],
                self.optimized_layer.weights,
                err_msg=f"Weights after backward do not match between layers (iter={iteration}).",
            )

            np.testing.assert_array_almost_equal(
                [neuron.threshold for neuron in self.layer.neurons],
                self.optimized_layer.thresholds,
                err_msg=f"Threshold after backward do not match between layers (iter={iteration}).",
            )

            self.layer.reset()
            self.optimized_layer.reset()


if __name__ == "__main__":
    unittest.main()
