import torch
import pytest

from spiking.learning.learner import Learner
from spiking.learning.stdp import STDP
from spiking.learning.wta import WinnerTakesAll
from spiking.threshold import NormalInitialization
from spiking.layers.integrate_and_fire import IntegrateAndFireLayer
from spiking.layers.sequential import SpikingSequential
from spiking.training import UnsupervisedTrainer


def make_layer(num_inputs=10, num_outputs=5):
    """Create a minimal IntegrateAndFireLayer for testing."""
    threshold_init = NormalInitialization(
        avg_threshold=5.0, min_threshold=1.0, std_dev=0.5
    )
    return IntegrateAndFireLayer(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        threshold_initialization=threshold_init,
        refractory_period=float("inf"),
    )


def make_stdp():
    return STDP(
        tau_pre=0.1,
        tau_post=0.1,
        max_pre_spike_time=1.0,
        learning_rate=0.1,
    )


def simulate_forward(layer, pre_spike_times):
    """Run a forward pass with given pre-spike times to produce post-spikes."""
    layer.train()
    layer.reset()
    # Feed spikes one timestep at a time
    for t_idx in range(len(pre_spike_times)):
        t = pre_spike_times[t_idx].item() if pre_spike_times[t_idx].isfinite() else None
        if t is not None:
            incoming = torch.zeros(layer.num_inputs)
            incoming[t_idx % layer.num_inputs] = 1.0
            layer.forward(incoming, current_time=t, dt=0.01)


class TestLearnerConstruction:
    def test_create_with_all_components(self):
        layer = make_layer()
        stdp = make_stdp()
        wta = WinnerTakesAll()
        learner = Learner(layer, stdp, competition=wta)
        assert learner.layer is layer
        assert learner.learning_mechanism is stdp
        assert learner.competition is wta

    def test_create_without_optional_components(self):
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        assert learner.competition is None
        assert learner.threshold_adaptation is None


class TestLearnerStep:
    def test_step_updates_weights_when_neurons_spike(self):
        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        layer.train()

        weights_before = layer.weights.detach().clone()

        # Manually set spike times to simulate a neuron having spiked
        layer._spike_times[0] = 0.5
        pre_spike_times = torch.rand(layer.num_inputs) * 0.3

        dw = learner.step(pre_spike_times)

        assert dw > 0.0, "Weight delta should be positive when learning occurs"
        assert not torch.equal(
            layer.weights[0], weights_before[0]
        ), "Weights of spiking neuron should have changed"

    def test_step_returns_zero_when_no_neurons_spike(self):
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        layer.train()

        # All spike times are inf (no spikes)
        pre_spike_times = torch.rand(layer.num_inputs)
        dw = learner.step(pre_spike_times)

        assert dw == 0.0

    def test_step_with_competition_only_updates_winner(self):
        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        wta = WinnerTakesAll()
        learner = Learner(layer, stdp, competition=wta)
        layer.train()

        weights_before = layer.weights.detach().clone()

        # Two neurons spiked, neuron 0 spiked first (winner)
        layer._spike_times[0] = 0.3
        layer._spike_times[1] = 0.5

        pre_spike_times = torch.rand(layer.num_inputs) * 0.2
        learner.step(pre_spike_times)

        # Winner's weights should change
        assert not torch.equal(layer.weights[0], weights_before[0])
        # Loser's weights should NOT change
        assert torch.equal(layer.weights[1], weights_before[1])

    def test_step_does_not_update_weights_in_eval_mode(self):
        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        layer.eval()

        weights_before = layer.weights.detach().clone()
        layer._spike_times[0] = 0.5
        pre_spike_times = torch.rand(layer.num_inputs) * 0.3

        dw = learner.step(pre_spike_times)

        # dw should still be computed (for monitoring) but weights not updated
        assert dw > 0.0
        assert torch.equal(
            layer.weights, weights_before
        ), "Weights should not change in eval mode"

    def test_step_updates_thresholds_when_adaptation_provided(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0
        )
        learner = Learner(layer, stdp, threshold_adaptation=adaptation)
        layer.train()

        thresholds_before = layer.thresholds.detach().clone()
        layer._spike_times[0] = 0.5
        pre_spike_times = torch.rand(layer.num_inputs) * 0.3

        learner.step(pre_spike_times)

        assert not torch.equal(
            layer.thresholds, thresholds_before
        ), "Thresholds should change when adaptation is provided"

    def test_step_does_not_update_thresholds_without_adaptation(self):
        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        layer.train()

        thresholds_before = layer.thresholds.detach().clone()
        layer._spike_times[0] = 0.5
        pre_spike_times = torch.rand(layer.num_inputs) * 0.3

        learner.step(pre_spike_times)

        assert torch.equal(
            layer.thresholds, thresholds_before
        ), "Thresholds should not change without adaptation"


class TestLearnerNeuronsToLearn:
    def test_neurons_to_learn_set_after_step(self):
        """step() should store the selected neurons on the learner instance."""
        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        wta = WinnerTakesAll()
        learner = Learner(layer, stdp, competition=wta)
        layer.train()

        layer._spike_times[0] = 0.3
        layer._spike_times[2] = 0.5
        pre_spike_times = torch.rand(layer.num_inputs) * 0.2

        learner.step(pre_spike_times)

        assert hasattr(learner, "neurons_to_learn")
        # WTA picks the earliest spiker (neuron 0)
        assert learner.neurons_to_learn.shape[0] == 1
        assert learner.neurons_to_learn.item() == 0

    def test_neurons_to_learn_without_competition(self):
        """Without competition, all spiking neurons should be selected."""
        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        layer.train()

        layer._spike_times[1] = 0.4
        layer._spike_times[3] = 0.6
        pre_spike_times = torch.rand(layer.num_inputs) * 0.3

        learner.step(pre_spike_times)

        assert hasattr(learner, "neurons_to_learn")
        indices = learner.neurons_to_learn.squeeze().tolist()
        assert 1 in indices
        assert 3 in indices


class TestLearnerLearningRateStep:
    def test_learning_rate_step_decays_learning_rate(self):
        layer = make_layer()
        stdp = STDP(
            tau_pre=0.1,
            tau_post=0.1,
            max_pre_spike_time=1.0,
            learning_rate=0.1,
            decay_factor=0.9,
        )
        learner = Learner(layer, stdp)

        learner.learning_rate_step()
        assert abs(stdp.learning_rate - 0.09) < 1e-6

    def test_learning_rate_step_decays_threshold_adaptation(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        layer = make_layer()
        stdp = make_stdp()
        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0, decay_factor=0.8
        )
        learner = Learner(layer, stdp, threshold_adaptation=adaptation)

        original_lr = adaptation.learning_rate
        learner.learning_rate_step()
        assert abs(adaptation.learning_rate - original_lr * 0.8) < 1e-6


class TestPreSpikeResolution:
    def test_single_layer_returns_input_spike_times(self):
        layer = make_layer()
        stdp = make_stdp()
        learner = Learner(layer, stdp)
        trainer = UnsupervisedTrainer(
            model=layer, learner=learner, image_shape=(1, 1, 10)
        )

        input_spike_times = torch.rand(10)
        result = trainer._get_pre_spike_times(input_spike_times)
        assert torch.equal(result, input_spike_times)

    def test_sequential_layer0_returns_input_spike_times(self):
        layer1 = make_layer(num_inputs=10, num_outputs=5)
        layer2 = make_layer(num_inputs=5, num_outputs=3)
        model = SpikingSequential(layer1, layer2)

        stdp = make_stdp()
        learner = Learner(layer1, stdp)
        trainer = UnsupervisedTrainer(
            model=model, learner=learner, image_shape=(1, 1, 10)
        )

        input_spike_times = torch.rand(10)
        result = trainer._get_pre_spike_times(input_spike_times)
        assert torch.equal(result, input_spike_times)

    def test_sequential_layer1_returns_layer0_spike_times(self):
        layer1 = make_layer(num_inputs=10, num_outputs=5)
        layer2 = make_layer(num_inputs=5, num_outputs=3)
        model = SpikingSequential(layer1, layer2)

        stdp = make_stdp()
        learner = Learner(layer2, stdp)
        trainer = UnsupervisedTrainer(
            model=model, learner=learner, image_shape=(1, 1, 10)
        )

        # Set layer1's spike times to known values
        layer1._spike_times = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        input_spike_times = torch.rand(10)
        result = trainer._get_pre_spike_times(input_spike_times)
        assert torch.equal(result, layer1.spike_times)

    def test_sequential_layer_not_found_raises(self):
        layer1 = make_layer(num_inputs=10, num_outputs=5)
        layer2 = make_layer(num_inputs=5, num_outputs=3)
        model = SpikingSequential(layer1)

        stdp = make_stdp()
        # Learner is bound to layer2, which isn't in the model
        learner = Learner(layer2, stdp)
        trainer = UnsupervisedTrainer(
            model=model, learner=learner, image_shape=(1, 1, 10)
        )

        with pytest.raises(ValueError):
            trainer._get_pre_spike_times(torch.rand(10))


class TestPlasticityBalanceAdaptation:
    def test_compute_balance_potentiation_dominant(self):
        """When pre spikes arrive before post spike, potentiation dominates."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )

        weights = torch.tensor([0.5, 0.5, 0.5])
        pre_spike_times = torch.tensor([0.1, 0.2, 0.3])  # all before post
        post_spike_time = 0.5

        balance = adaptation.compute_balance(weights, pre_spike_times, post_spike_time)
        assert balance > 0, "Balance should be positive when pre spikes precede post"

    def test_compute_balance_depression_dominant(self):
        """When pre spikes arrive after post spike, depression dominates."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )

        weights = torch.tensor([0.5, 0.5, 0.5])
        pre_spike_times = torch.tensor([0.7, 0.8, 0.9])  # all after post
        post_spike_time = 0.5

        balance = adaptation.compute_balance(weights, pre_spike_times, post_spike_time)
        assert balance < 0, "Balance should be negative when pre spikes follow post"

    def test_update_adjusts_thresholds_for_spiking_neurons(self):
        """update() should only modify thresholds of neurons that spiked."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )

        current_thresholds = torch.tensor([5.0, 5.0, 5.0])
        spike_times = torch.tensor([0.5, float("inf"), float("inf")])
        weights = torch.rand(3, 10)
        pre_spike_times = torch.rand(10) * 0.3

        updated = adaptation.update(
            current_thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        # Neuron 0 spiked, so its threshold should change
        assert updated[0] != current_thresholds[0]
        # Neurons 1,2 didn't spike, thresholds unchanged
        assert updated[1] == current_thresholds[1]
        assert updated[2] == current_thresholds[2]

    def test_potentiation_dominant_decreases_threshold(self):
        """When potentiation dominates, threshold should decrease (negative feedback)."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )

        current_thresholds = torch.tensor([50.0])
        spike_times = torch.tensor([0.5])
        weights = torch.ones(1, 5)
        pre_spike_times = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.45])  # all before post

        updated = adaptation.update(
            current_thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        assert updated[0] < 50.0, "Potentiation dominant → threshold should decrease"

    def test_depression_dominant_increases_threshold(self):
        """When depression dominates, threshold should increase (negative feedback)."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )

        current_thresholds = torch.tensor([50.0])
        spike_times = torch.tensor([0.5])
        weights = torch.ones(1, 5)
        pre_spike_times = torch.tensor([0.7, 0.8, 0.9, 0.95, 0.99])  # all after post

        updated = adaptation.update(
            current_thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        assert updated[0] > 50.0, "Depression dominant → threshold should increase"

    def test_update_clamps_thresholds(self):
        """Thresholds should be clamped within [min_threshold, max_threshold]."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=1000.0,  # huge lr to push past bounds
            min_threshold=2.0,
            max_threshold=10.0,
        )

        current_thresholds = torch.tensor([5.0])
        spike_times = torch.tensor([0.5])
        weights = torch.ones(1, 5)
        pre_spike_times = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.45])

        updated = adaptation.update(
            current_thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        assert updated[0] >= 2.0
        assert updated[0] <= 10.0

    def test_learning_rate_step_decays(self):
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
            decay_factor=0.9,
        )

        adaptation.learning_rate_step()
        assert abs(adaptation.learning_rate - 0.09) < 1e-6

    def test_sign_only_uses_fixed_step_size(self):
        """With sign_only=True, the update magnitude should equal learning_rate regardless of balance magnitude."""
        from spiking.threshold import PlasticityBalanceAdaptation

        lr = 0.1
        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=lr,
            min_threshold=1.0,
            max_threshold=100.0,
            sign_only=True,
        )

        current_thresholds = torch.tensor([50.0])
        spike_times = torch.tensor([0.5])
        weights = torch.ones(1, 5)
        pre_spike_times = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.45])

        updated = adaptation.update(
            current_thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        # All pre spikes before post → positive balance → threshold decreases by exactly lr
        assert abs(updated[0].item() - (50.0 - lr)) < 1e-5

    def test_sign_only_negative_direction(self):
        """sign_only=True with depression-dominant balance should increase by exactly lr."""
        from spiking.threshold import PlasticityBalanceAdaptation

        lr = 0.1
        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=lr,
            min_threshold=1.0,
            max_threshold=100.0,
            sign_only=True,
        )

        current_thresholds = torch.tensor([50.0])
        spike_times = torch.tensor([0.5])
        weights = torch.ones(1, 5)
        pre_spike_times = torch.tensor([0.7, 0.8, 0.9, 0.95, 0.99])

        updated = adaptation.update(
            current_thresholds,
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        assert abs(updated[0].item() - (50.0 + lr)) < 1e-5

    def test_sign_only_defaults_false(self):
        """sign_only should default to False for backward compatibility."""
        from spiking.threshold import PlasticityBalanceAdaptation

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0, learning_rate=0.1, min_threshold=1.0, max_threshold=100.0,
        )
        assert adaptation.sign_only is False

    def test_vectorized_update_matches_per_neuron_loop(self):
        """Vectorized update() must match per-neuron compute_balance() loop."""
        from spiking.threshold import PlasticityBalanceAdaptation

        torch.manual_seed(123)
        num_neurons = 8
        num_inputs = 20

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )

        current_thresholds = torch.rand(num_neurons) * 10 + 5
        weights = torch.rand(num_neurons, num_inputs)
        pre_spike_times = torch.rand(num_inputs) * 0.8
        # Some neurons spiked, some didn't
        spike_times = torch.full((num_neurons,), float("inf"))
        spike_times[0] = 0.3
        spike_times[2] = 0.5
        spike_times[4] = 0.7
        spike_times[6] = 0.1

        # Reference: per-neuron loop using compute_balance()
        expected_deltas = torch.zeros_like(current_thresholds)
        spiked_mask = torch.isfinite(spike_times)
        spiked_indices = torch.nonzero(spiked_mask, as_tuple=True)[0]
        for neuron_idx in spiked_indices:
            idx = neuron_idx.item()
            balance = adaptation.compute_balance(
                weights[idx], pre_spike_times, spike_times[idx].item()
            )
            expected_deltas[idx] = adaptation.learning_rate * balance
        expected = torch.clamp(
            current_thresholds - expected_deltas,
            adaptation.min_threshold,
            adaptation.max_threshold,
        )

        # Actual: vectorized update()
        actual = adaptation.update(
            current_thresholds.clone(),
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_vectorized_update_sign_only_matches_loop(self):
        """Vectorized update() with sign_only=True must match per-neuron loop."""
        from spiking.threshold import PlasticityBalanceAdaptation

        torch.manual_seed(456)
        num_neurons = 6
        num_inputs = 15

        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
            sign_only=True,
        )

        current_thresholds = torch.rand(num_neurons) * 10 + 5
        weights = torch.rand(num_neurons, num_inputs)
        pre_spike_times = torch.rand(num_inputs) * 0.8
        spike_times = torch.full((num_neurons,), float("inf"))
        spike_times[1] = 0.4
        spike_times[3] = 0.6
        spike_times[5] = 0.2

        # Reference: per-neuron loop
        expected_deltas = torch.zeros_like(current_thresholds)
        spiked_mask = torch.isfinite(spike_times)
        spiked_indices = torch.nonzero(spiked_mask, as_tuple=True)[0]
        for neuron_idx in spiked_indices:
            idx = neuron_idx.item()
            balance = adaptation.compute_balance(
                weights[idx], pre_spike_times, spike_times[idx].item()
            )
            direction = 1.0 if balance > 0 else (-1.0 if balance < 0 else 0.0)
            expected_deltas[idx] = adaptation.learning_rate * direction
        expected = torch.clamp(
            current_thresholds - expected_deltas,
            adaptation.min_threshold,
            adaptation.max_threshold,
        )

        actual = adaptation.update(
            current_thresholds.clone(),
            spike_times,
            weights=weights,
            pre_spike_times=pre_spike_times,
        )

        assert torch.allclose(actual, expected, atol=1e-6)

    def test_through_learner_step(self):
        """PlasticityBalanceAdaptation works end-to-end through Learner.step()."""
        from spiking.threshold import PlasticityBalanceAdaptation

        torch.manual_seed(42)
        layer = make_layer()
        adaptation = PlasticityBalanceAdaptation(
            tau=20.0,
            learning_rate=0.1,
            min_threshold=1.0,
            max_threshold=100.0,
        )
        learner = Learner(
            layer, learning_mechanism=None, threshold_adaptation=adaptation
        )
        layer.train()

        thresholds_before = layer.thresholds.detach().clone()
        weights_before = layer.weights.detach().clone()
        layer._spike_times[0] = 0.5

        pre_spike_times = torch.rand(layer.num_inputs) * 0.3
        dw = learner.step(pre_spike_times)

        # No learning mechanism → no weight changes
        assert torch.equal(layer.weights, weights_before)
        assert dw == 0.0
        # Threshold should have changed for neuron 0
        assert not torch.equal(layer.thresholds, thresholds_before)


class TestLearnerOptionalLearningMechanism:
    def test_create_with_no_learning_mechanism(self):
        layer = make_layer()
        learner = Learner(layer, learning_mechanism=None)
        assert learner.learning_mechanism is None

    def test_step_no_learning_no_adaptation_returns_zero(self):
        layer = make_layer()
        learner = Learner(layer, learning_mechanism=None)
        layer.train()
        layer._spike_times[0] = 0.5

        pre_spike_times = torch.rand(layer.num_inputs)
        dw = learner.step(pre_spike_times)

        assert dw == 0.0

    def test_learning_rate_step_with_no_learning_mechanism(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        layer = make_layer()
        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0, decay_factor=0.8
        )
        learner = Learner(
            layer, learning_mechanism=None, threshold_adaptation=adaptation
        )

        original_lr = adaptation.learning_rate
        learner.learning_rate_step()
        assert abs(adaptation.learning_rate - original_lr * 0.8) < 1e-6


class TestUnsupervisedTrainerCallback:
    def test_trainer_calls_on_batch_end(self):
        """UnsupervisedTrainer should call on_batch_end after each step_batch."""
        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        calls = []
        trainer = UnsupervisedTrainer(
            model=layer,
            learner=learner,
            image_shape=(1, 1, 10),
            on_batch_end=lambda batch_idx, dw, split: calls.append(
                (batch_idx, dw, split)
            ),
        )

        times = torch.rand(1, 1, 10) * 0.5

        trainer.step_batch(0, times, "0", split="train")

        assert len(calls) == 1
        assert calls[0][0] == 0
        assert calls[0][2] == "train"

    def test_trainer_works_without_callback(self):
        """UnsupervisedTrainer should work fine with on_batch_end=None."""
        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        trainer = UnsupervisedTrainer(
            model=layer,
            learner=learner,
            image_shape=(1, 1, 10),
        )
        assert trainer.on_batch_end is None


class TestUnsupervisedTrainerEarlyStopping:
    def test_early_stopping_false_processes_all_timesteps(self):
        """With early_stopping=False, the trainer should NOT break on first spike."""
        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        trainer = UnsupervisedTrainer(
            model=layer,
            learner=learner,
            image_shape=(1, 1, 10),
            early_stopping=False,
        )
        # Just verify it accepts the flag without error
        assert trainer.early_stopping is False

    def test_early_stopping_defaults_true(self):
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        trainer = UnsupervisedTrainer(
            model=layer,
            learner=learner,
            image_shape=(1, 1, 10),
        )
        assert trainer.early_stopping is True


class TestTrainFunction:
    def test_train_runs_correct_epochs(self):
        """train() should call step_epoch for each epoch."""
        from spiking.training import train
        from unittest.mock import MagicMock

        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter([]))
        mock_loader.__len__ = MagicMock(return_value=0)

        initial_lr = stdp.learning_rate
        decay = 0.9
        stdp.decay_factor = decay

        train(
            layer, learner, mock_loader, num_epochs=3,
            image_shape=(1, 1, 10), progress=False,
        )

        expected_lr = initial_lr * (decay ** 3)
        assert abs(stdp.learning_rate - expected_lr) < 1e-6

    def test_train_calls_callback(self):
        """train() should wire on_batch_end through to UnsupervisedTrainer."""
        from spiking.training import train
        from unittest.mock import MagicMock

        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        times = torch.rand(1, 1, 10) * 0.5
        mock_loader = [(times, "0")]

        calls = []
        train(
            layer, learner, mock_loader, num_epochs=1,
            image_shape=(1, 1, 10),
            on_batch_end=lambda batch_idx, dw, split: calls.append(split),
            progress=False,
        )

        assert "train" in calls

    def test_train_without_val_loader(self):
        """train() should work without a validation loader."""
        from spiking.training import train
        from unittest.mock import MagicMock

        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter([]))
        mock_loader.__len__ = MagicMock(return_value=0)

        # Should not raise
        train(
            layer, learner, mock_loader, num_epochs=1,
            image_shape=(1, 1, 10), progress=False,
        )

    def test_train_calls_on_epoch_end(self):
        """train() should call on_epoch_end with (epoch, num_epochs) after each epoch."""
        from spiking.training import train
        from unittest.mock import MagicMock

        torch.manual_seed(42)
        layer = make_layer(num_inputs=10, num_outputs=5)
        stdp = make_stdp()
        learner = Learner(layer, stdp)

        mock_loader = MagicMock()
        mock_loader.__iter__ = MagicMock(return_value=iter([]))
        mock_loader.__len__ = MagicMock(return_value=0)

        epoch_calls = []
        train(
            layer, learner, mock_loader, num_epochs=3,
            image_shape=(1, 1, 10), progress=False,
            on_epoch_end=lambda epoch, total: epoch_calls.append((epoch, total)),
        )

        assert epoch_calls == [(1, 3), (2, 3), (3, 3)]


class TestSequentialThresholdAdaptation:
    def test_chains_multiple_adaptations(self):
        """Both adaptations are applied in order."""
        from spiking.threshold import (
            TargetTimestampAdaptation,
            CompetitiveThresholdAdaptation,
            SequentialThresholdAdaptation,
        )

        adaptation1 = TargetTimestampAdaptation(
            min_threshold=1.0, target_timestamp=0.5, learning_rate=1.0
        )
        adaptation2 = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0
        )
        seq = SequentialThresholdAdaptation([adaptation1, adaptation2])

        thresholds = torch.tensor([5.0, 5.0, 5.0])
        spike_times = torch.tensor([0.3, float("inf"), float("inf")])
        neurons_to_learn = torch.tensor([0])

        # Apply adaptation1 alone
        after_first = adaptation1.update(thresholds.clone(), spike_times)
        # Then apply adaptation2 to that result
        expected = adaptation2.update(
            after_first, spike_times, neurons_to_learn=neurons_to_learn
        )
        # Sequential should produce the same result
        result = seq.update(
            thresholds.clone(), spike_times, neurons_to_learn=neurons_to_learn
        )
        assert torch.allclose(result, expected)

    def test_learning_rate_step_calls_all(self):
        """learning_rate_step propagates to all children."""
        from spiking.threshold import (
            TargetTimestampAdaptation,
            CompetitiveThresholdAdaptation,
            SequentialThresholdAdaptation,
        )

        adaptation1 = TargetTimestampAdaptation(
            min_threshold=1.0, target_timestamp=0.5, learning_rate=1.0, decay_factor=0.5
        )
        adaptation2 = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0, decay_factor=0.9
        )
        seq = SequentialThresholdAdaptation([adaptation1, adaptation2])

        seq.learning_rate_step()

        assert abs(adaptation1.learning_rate - 0.5) < 1e-6
        assert abs(adaptation2.learning_rate - 4.5) < 1e-6

    def test_single_adaptation_works(self):
        """Edge case: list of 1 behaves like the inner adaptation."""
        from spiking.threshold import (
            CompetitiveThresholdAdaptation,
            SequentialThresholdAdaptation,
        )

        inner = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0
        )
        seq = SequentialThresholdAdaptation([inner])

        thresholds = torch.tensor([5.0, 5.0])
        spike_times = torch.tensor([0.3, float("inf")])
        neurons_to_learn = torch.tensor([0])

        expected = inner.update(
            thresholds.clone(), spike_times, neurons_to_learn=neurons_to_learn
        )
        result = seq.update(
            thresholds.clone(), spike_times, neurons_to_learn=neurons_to_learn
        )
        assert torch.allclose(result, expected)

    def test_through_learner_step(self):
        """End-to-end via Learner."""
        from spiking.threshold import (
            TargetTimestampAdaptation,
            CompetitiveThresholdAdaptation,
            SequentialThresholdAdaptation,
        )

        torch.manual_seed(42)
        layer = make_layer()
        stdp = make_stdp()
        seq = SequentialThresholdAdaptation([
            TargetTimestampAdaptation(
                min_threshold=1.0, target_timestamp=0.5, learning_rate=1.0
            ),
            CompetitiveThresholdAdaptation(
                min_threshold=1.0, learning_rate=5.0
            ),
        ])
        learner = Learner(layer, stdp, competition=WinnerTakesAll(),
                          threshold_adaptation=seq)
        layer.train()

        thresholds_before = layer.thresholds.detach().clone()
        layer._spike_times[0] = 0.3
        pre_spike_times = torch.rand(layer.num_inputs) * 0.2

        learner.step(pre_spike_times)

        assert not torch.equal(layer.thresholds, thresholds_before)


class TestCompetitiveThresholdAdaptation:
    def test_winners_thresholds_increase(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0
        )
        thresholds = torch.tensor([10.0, 10.0, 10.0])
        spike_times = torch.tensor([0.3, 0.5, float("inf")])
        neurons_to_learn = torch.tensor([0])

        updated = adaptation.update(
            thresholds.clone(), spike_times, neurons_to_learn=neurons_to_learn
        )
        assert updated[0] > thresholds[0]

    def test_losers_thresholds_decrease(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0
        )
        thresholds = torch.tensor([10.0, 10.0, 10.0])
        spike_times = torch.tensor([0.3, 0.5, float("inf")])
        neurons_to_learn = torch.tensor([0])

        updated = adaptation.update(
            thresholds.clone(), spike_times, neurons_to_learn=neurons_to_learn
        )
        # Losers (neurons 1 and 2) should decrease
        assert updated[1] < thresholds[1]
        assert updated[2] < thresholds[2]

    def test_respects_min_threshold(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=9.0, learning_rate=100.0
        )
        thresholds = torch.tensor([10.0, 10.0])
        spike_times = torch.tensor([0.3, float("inf")])
        neurons_to_learn = torch.tensor([0])

        updated = adaptation.update(
            thresholds.clone(), spike_times, neurons_to_learn=neurons_to_learn
        )
        assert updated.min() >= 9.0

    def test_requires_neurons_to_learn(self):
        from spiking.threshold import CompetitiveThresholdAdaptation

        adaptation = CompetitiveThresholdAdaptation(
            min_threshold=1.0, learning_rate=5.0
        )
        thresholds = torch.tensor([10.0, 10.0])
        spike_times = torch.tensor([0.3, float("inf")])

        with pytest.raises(ValueError):
            adaptation.update(thresholds, spike_times)


class TestTargetTimestampAdaptation:
    def test_early_spikes_increase_thresholds(self):
        """Spikes before target_timestamp should increase thresholds."""
        from spiking.threshold import TargetTimestampAdaptation

        adaptation = TargetTimestampAdaptation(
            min_threshold=1.0, target_timestamp=0.5, learning_rate=1.0
        )
        thresholds = torch.tensor([10.0, 10.0])
        # Neuron 0 spikes at 0.2 (before target 0.5)
        spike_times = torch.tensor([0.2, float("inf")])

        updated = adaptation.update(thresholds.clone(), spike_times)
        # delta_t = spike_time - target = 0.2 - 0.5 = -0.3 (negative)
        # threshold_delta = lr * delta_t = 1.0 * -0.3 = -0.3
        # updated = current - threshold_delta = 10.0 - (-0.3) = 10.3
        assert updated[0] > thresholds[0]

    def test_late_spikes_decrease_thresholds(self):
        """Spikes after target_timestamp should decrease thresholds."""
        from spiking.threshold import TargetTimestampAdaptation

        adaptation = TargetTimestampAdaptation(
            min_threshold=1.0, target_timestamp=0.5, learning_rate=1.0
        )
        thresholds = torch.tensor([10.0, 10.0])
        # Neuron 0 spikes at 0.8 (after target 0.5)
        spike_times = torch.tensor([0.8, float("inf")])

        updated = adaptation.update(thresholds.clone(), spike_times)
        assert updated[0] < thresholds[0]

    def test_non_spiking_neurons_unchanged(self):
        from spiking.threshold import TargetTimestampAdaptation

        adaptation = TargetTimestampAdaptation(
            min_threshold=1.0, target_timestamp=0.5, learning_rate=1.0
        )
        thresholds = torch.tensor([10.0, 10.0])
        spike_times = torch.tensor([0.3, float("inf")])

        updated = adaptation.update(thresholds.clone(), spike_times)
        assert updated[1] == thresholds[1]

    def test_respects_min_threshold(self):
        from spiking.threshold import TargetTimestampAdaptation

        adaptation = TargetTimestampAdaptation(
            min_threshold=5.0, target_timestamp=0.1, learning_rate=100.0
        )
        thresholds = torch.tensor([6.0])
        # Spike very late → large decrease
        spike_times = torch.tensor([0.9])

        updated = adaptation.update(thresholds.clone(), spike_times)
        assert updated[0] >= 5.0
