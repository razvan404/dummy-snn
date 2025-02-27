import numpy as np
import unittest

from ..learning import STDP
from ..competition import WinnerTakesAll
from ..layers import IntegrateAndFireLayer, IntegrateAndFireOptimizedLayer


class TestOptimizedIntegrateAndFireLayer(unittest.TestCase):
    def setUp(self):
        self.num_inputs = 10
        self.num_outputs = 5
        self.learning_mechanism = STDP()
        self.competition_mechanism = WinnerTakesAll()
        self.threshold = 1.0
        self.refractory_period = np.inf
        self.min_threshold = 0.2
        # self.threshold_initialization = NormalInitialization(self.min_threshold)
        # self.threshold_adaptation = FalezAdaptation(
        #     self.min_threshold, threshold_learning_rate=2e-2, target_timestamp=0.7
        # )

        self.layer = IntegrateAndFireLayer(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            learning_mechanism=self.learning_mechanism,
            competition_mechanism=self.competition_mechanism,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            # threshold_adaptation=self.threshold_adaptation,
            # threshold_initialization=self.threshold_initialization,
        )

        self.optimized_layer = IntegrateAndFireOptimizedLayer(
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            learning_mechanism=self.learning_mechanism,
            competition_mechanism=self.competition_mechanism,
            threshold=self.threshold,
            refractory_period=self.refractory_period,
            # threshold_adaptation=self.threshold_adaptation,
            # threshold_initialization=self.threshold_initialization,
        )

        self.optimized_layer.weights = np.array(
            [neuron.weights for neuron in self.layer.neurons]
        )

    def test_layer_equivalence(self):
        incoming_spikes = np.random.randint(0, 2, size=(self.num_inputs,)).astype(
            np.float32
        )
        target_time = 0.3
        dt = 0.05
        target_spike_times = np.random.uniform(
            target_time - 0.1, target_time + 0.1, size=(self.num_inputs,)
        )
        target_spike_times[incoming_spikes == 0.0] = np.inf
        print("Incoming spikes:", incoming_spikes)
        print("Target spike times:", target_spike_times)

        for iteration in range(20):
            print(f"Iteration {iteration}")
            np.testing.assert_array_almost_equal(
                [neuron.weights for neuron in self.layer.neurons],
                self.optimized_layer.weights,
                err_msg=f"Weights do not match between layers (iter={iteration}).",
            )

            num_iter_layer = 0
            current_time_layer = 0
            while (
                not np.any(
                    spikes_layer := self.layer.forward(
                        incoming_spikes, current_time_layer, dt
                    )
                )
                and not current_time_layer < 10
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
                and not current_time_layer < 10
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

            np.testing.assert_array_almost_equal(
                [neuron.weights for neuron in self.layer.neurons],
                self.optimized_layer.weights,
                err_msg=f"Weights after backward do not match between layers (iter={iteration}).",
            )


if __name__ == "__main__":
    unittest.main()
