# Conv SNN Experiment Pipeline

Three scripts for convolutional integrate-and-fire SNN experiments:
train, evaluate (LinearSVC + Ridge), and iterative threshold optimization.

All commands run from the project root.

## Directory layout

```
logs/{dataset}/sweep/nf_{N}/tobj_{T}/seed_{S}/
    model.pth                                    # trained model
    setup.json                                   # hyperparameters
    training_logs.pt                             # per-epoch stats
    metrics.json                                 # LinearSVC + Ridge accuracies
    iterative_optimization_{ordering}.json       # optimization results
    iterative_optimization_{ordering}_convergence.png
```

`{dataset}` is `cifar10_whitened` for CIFAR-10, `mnist` for MNIST.

## Common arguments

All three scripts accept:

| Argument         | Default    | Description                        |
|------------------|------------|------------------------------------|
| `--dataset`      | `cifar10`  | `mnist` or `cifar10`               |
| `--num-filters`  | paper      | Number of conv filters             |
| `--t-obj`        | paper      | Target timestamp for STDP training |
| `--seed`         | `1`        | Random seed                        |
| `--force`        | off        | Overwrite existing results         |

Paper defaults: MNIST (32 filters, t_obj=0.75), CIFAR-10 (256 filters, t_obj=0.97).

## 1. Train

```bash
python -m applications.conv_threshold_optimization.train \
    --dataset cifar10 --num-filters 256 --t-obj 0.97 --seed 1
```

Trains a single conv SNN layer with STDP on random patches.

## 2. Evaluate

```bash
python -m applications.conv_threshold_optimization.evaluate \
    --dataset cifar10 --num-filters 256 --t-obj 0.97 --seed 1 \
    --device cuda
```

Extracts conv features with sum pooling, fits both LinearSVC and Ridge,
saves `metrics.json`. Uses GPU-accelerated classifiers (cuml) automatically
when the `cuda` extra is installed.

## 3. Optimize

```bash
python -m applications.conv_threshold_optimization.optimize \
    --dataset cifar10 --num-filters 256 --t-obj 0.97 --seed 1 \
    --device cuda --ordering descending_importance \
    --classifier svc --num-rounds 25 --step-size 0.2
```

Runs iterative coordinate descent for a single ordering strategy.
Each round: extract features at current thresholds +/- step, fit the
classifier, greedily optimize each filter via column swaps. Repeat until
convergence.

### Ordering strategies

12 strategies organized in four families. Hybrid variants interleave from
both ends of the sorted order (most → least → 2nd most → 2nd least → ...).

**Classifier-aware** — based on the fitted classifier's weight magnitudes:

| Ordering                 | Description                                  |
|--------------------------|----------------------------------------------|
| `descending_importance`  | Highest classifier coefficient magnitude first |
| `ascending_importance`   | Lowest importance first                       |
| `hybrid_importance`      | Interleaved: most important ↔ least important |

**Inference spike time** — based on mean spike time over the current features:

| Ordering                 | Description                                  |
|--------------------------|----------------------------------------------|
| `early_spike`            | Earliest mean spike time first                |
| `late_spike`             | Latest mean spike time first                  |
| `hybrid_spike_time`      | Interleaved: earliest ↔ latest                |

**Threshold deviation** — based on |threshold_i − mean(thresholds)|:

| Ordering                 | Description                                  |
|--------------------------|----------------------------------------------|
| `high_abs_drift`         | Largest deviation from mean threshold first   |
| `low_abs_drift`          | Smallest deviation first                      |
| `hybrid_abs_drift`       | Interleaved: most deviated ↔ least deviated   |

**Training spike time** — based on mean spike time recorded during STDP
training (requires `mean_spike_time_per_neuron` in `training_metrics.json`;
filters that never fired are placed last):

| Ordering                 | Description                                  |
|--------------------------|----------------------------------------------|
| `training_early_spike`   | Earliest training-time spike first            |
| `training_late_spike`    | Latest training-time spike first              |
| `hybrid_training_spike`  | Interleaved: earliest ↔ latest training spike |

## Full pipeline example

```bash
# Train
python -m applications.conv_threshold_optimization.train --dataset cifar10 --seed 1

# Evaluate
python -m applications.conv_threshold_optimization.evaluate --dataset cifar10 --seed 1 --device cuda

# Optimize with different orderings
for ord in descending_importance early_spike hybrid_importance; do
    python -m applications.conv_threshold_optimization.optimize --dataset cifar10 --seed 1 \
        --device cuda --ordering $ord
done
```

## Utility modules

| Module                                          | Purpose                            |
|-------------------------------------------------|------------------------------------|
| `common.py`                                     | Shared helpers, path resolution    |
| `paper_hyperparams.py`                          | Paper-exact hyperparameter configs |
| `datasets/`                                     | Dataset loading and preprocessing  |
| `conv_learning/train.py`                        | Core training loop                 |
| `threshold_research/iterative_optimization.py`  | Coordinate descent optimizer       |
| `threshold_research/filter_ordering.py`         | 12 filter ordering strategies      |
| `threshold_research/conv_neuron_perturbation.py`| Multi-threshold conv accumulation  |
| `threshold_research/perturbation_params.py`     | Perturbation fraction config       |
