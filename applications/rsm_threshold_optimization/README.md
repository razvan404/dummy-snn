# RSM Threshold Optimization

Response Surface Methodology (RSM) for jointly optimizing all SNN thresholds.

## Approach

Instead of optimizing one neuron at a time (sequential greedy) or using noisy gradient estimates (SPSA/STE), RSM:

1. **Design**: Generate N random threshold configurations (Rademacher ±1 design)
2. **Evaluate**: Use multi-threshold inference (one conv2d pass, N threshold checks) to get accuracy for each
3. **Fit**: Regress accuracy on the design matrix: `acc ≈ β₀ + Σ βᵢδᵢ`
4. **Optimize**: Set each threshold in the direction of its main effect βᵢ

## Key Assumptions

- **Linearity**: The accuracy response to small threshold perturbations is approximately linear. Validated by the perturbation sweep showing smooth accuracy-vs-fraction curves for most neurons.
- **Additivity (main effects only)**: The effect of neuron i's threshold is independent of neuron j's. This is approximately true because neurons don't interact during inference (no lateral inhibition). They only interact through the downstream classifier.
- **Perturbation scale matters**: Too large → nonlinear effects dominate, model is inaccurate. Too small → signal lost in noise. Default 5% is a balance.
- **Classifier variance**: Refitting the classifier (Ridge/SVC) introduces noise (~0.1-0.3% accuracy variance). With 300 configurations, the regression averages this out.

## Comparison with Other Approaches

| Method | Evals | Captures interactions | Speed |
|--------|-------|----------------------|-------|
| Sequential greedy | 256 × 25 | Partial (via Woodbury) | ~30 min |
| SPSA (10 steps) | 30 | Implicit (noisy) | ~75 min |
| **RSM (main effects)** | **~6 forward passes** | **Explicit (βᵢ)** | **~15 min** |
| RSM (with interactions) | ~6 forward passes | **Explicit (βᵢⱼ)** | ~15 min |

## Usage

```bash
# Main effects only (fast, requires n_configs >= num_filters)
python -m applications.rsm_threshold_optimization \
    --n-configs 300 --perturbation-scale 0.05 --device cuda

# With pairwise interactions (needs more configs for stable fit)
python -m applications.rsm_threshold_optimization \
    --n-configs 500 --fit-interactions --device cuda

# Using SVC instead of Ridge for evaluation
python -m applications.rsm_threshold_optimization \
    --classifier svc --n-configs 300 --device cuda
```

## Output

- `results.json`: Baseline accuracy, RSM-predicted optimal accuracy, main effects, R²
- `rsm_analysis.png`: Diagnostic plots (predicted vs actual, top effects, accuracy distribution)

## Parameters

- `--n-configs`: Number of random threshold configurations to evaluate. More = better fit but slower. Minimum ~p+10 for main effects, ~p(p+1)/2 for interactions.
- `--perturbation-scale`: Each threshold is perturbed by ±scale × original value. Default 0.05 (5%).
- `--fit-interactions`: Include pairwise βᵢⱼ terms. Requires more configs and regularization.
- `--classifier`: "ridge" (fast, ~1s/eval) or "svc" (slower, ~30s/eval but often more accurate).
- `--config-batch-size`: How many configs to evaluate per forward pass. Limited by GPU memory.
- `--chunk-size`: Images per chunk in multi-threshold inference. Smaller = less memory.
