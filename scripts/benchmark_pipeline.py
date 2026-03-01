import time
import torch

from spiking import (
    IntegrateAndFireLayer,
    Learner,
    STDP,
    WinnerTakesAll,
    CompetitiveThresholdAdaptation,
    NormalInitialization,
    iterate_spikes,
)
from spiking.training import UnsupervisedTrainer


NUM_INPUTS = 512
NUM_OUTPUTS = 1024
NUM_SAMPLES = 200
NUM_WARMUP = 5


def make_layer():
    threshold_init = NormalInitialization(
        avg_threshold=156.8, min_threshold=1.0, std_dev=1.0
    )
    return IntegrateAndFireLayer(
        num_inputs=NUM_INPUTS,
        num_outputs=NUM_OUTPUTS,
        threshold_initialization=threshold_init,
        refractory_period=float("inf"),
    )


def make_learner(layer):
    return Learner(
        layer,
        learning_mechanism=STDP(
            tau_pre=0.1,
            tau_post=0.1,
            max_pre_spike_time=1.0,
            learning_rate=0.1,
        ),
        competition=WinnerTakesAll(),
        threshold_adaptation=CompetitiveThresholdAdaptation(
            min_threshold=1.0,
            learning_rate=5.0,
        ),
    )


def make_spike_times():
    """Generate latency-encoded spike times similar to MNIST preprocessing."""
    times = torch.full((NUM_INPUTS,), float("inf"))
    num_spikes = torch.randint(50, 200, (1,)).item()
    spike_indices = torch.randperm(NUM_INPUTS)[:num_spikes]
    times[spike_indices] = torch.rand(num_spikes) * 0.9 + 0.01
    return times


def benchmark_iterate_spikes(samples):
    """Time iterate_spikes: frame generation from spike times."""
    elapsed = 0.0
    for times in samples:
        t0 = time.perf_counter()
        for _ in iterate_spikes(times):
            pass
        elapsed += time.perf_counter() - t0
    return elapsed


def benchmark_forward(layer, samples):
    """Time layer.forward() calls (the full forward pass per sample)."""
    elapsed = 0.0
    layer.eval()
    with torch.no_grad():
        for times in samples:
            layer.reset()
            t0 = time.perf_counter()
            for incoming_spikes, current_time, dt in iterate_spikes(times):
                layer.forward(incoming_spikes, current_time, dt)
            elapsed += time.perf_counter() - t0
    return elapsed


def benchmark_learner_step(layer, learner, samples):
    """Time learner.step() after a forward pass."""
    elapsed = 0.0
    layer.train()
    with torch.no_grad():
        for times in samples:
            layer.reset()
            for incoming_spikes, current_time, dt in iterate_spikes(times):
                output_spikes = layer.forward(incoming_spikes, current_time, dt)
                if torch.any(output_spikes == 1.0):
                    break
            t0 = time.perf_counter()
            learner.step(times)
            elapsed += time.perf_counter() - t0
    return elapsed


def benchmark_step_batch(layer, learner, samples):
    """Time the full step_batch (iterate + forward + learn)."""
    trainer = UnsupervisedTrainer(
        layer,
        learner,
        image_shape=(1, 1, NUM_INPUTS),
    )
    layer.train()
    elapsed = 0.0
    for i, times in enumerate(samples):
        t0 = time.perf_counter()
        trainer.step_batch(i, times.unsqueeze(0), None, split="train")
        elapsed += time.perf_counter() - t0
    return elapsed


def run():
    torch.manual_seed(42)
    samples = [make_spike_times() for _ in range(NUM_SAMPLES + NUM_WARMUP)]
    warmup_samples = samples[:NUM_WARMUP]
    bench_samples = samples[NUM_WARMUP:]

    # Warmup
    layer = make_layer()
    learner = make_learner(layer)
    for times in warmup_samples:
        layer.reset()
        for incoming_spikes, current_time, dt in iterate_spikes(times):
            layer.forward(incoming_spikes, current_time, dt)
        learner.step(times)

    print(f"Config: {NUM_INPUTS} inputs, {NUM_OUTPUTS} outputs, {NUM_SAMPLES} samples")
    print(f"{'Component':<30} {'Total (s)':>10} {'Per sample (ms)':>15}")
    print("-" * 58)

    t = benchmark_iterate_spikes(bench_samples)
    print(f"{'iterate_spikes':<30} {t:>10.3f} {t/NUM_SAMPLES*1000:>15.2f}")

    layer = make_layer()
    t = benchmark_forward(layer, bench_samples)
    print(f"{'forward pass':<30} {t:>10.3f} {t/NUM_SAMPLES*1000:>15.2f}")

    layer = make_layer()
    learner = make_learner(layer)
    t = benchmark_learner_step(layer, learner, bench_samples)
    print(f"{'learner.step':<30} {t:>10.3f} {t/NUM_SAMPLES*1000:>15.2f}")

    layer = make_layer()
    learner = make_learner(layer)
    t = benchmark_step_batch(layer, learner, bench_samples)
    print(f"{'step_batch (full)':<30} {t:>10.3f} {t/NUM_SAMPLES*1000:>15.2f}")


if __name__ == "__main__":
    run()
