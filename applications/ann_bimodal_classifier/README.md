# ANN Bimodal Classifier

Same bounded encoder + bimodal regularizer as `ann_bimodal_autoencoder`,
but downstream task is CIFAR-10 classification:

```
input → Conv2d (clipped to [0,1], regularized) → ReLU → BatchNorm
      → AdaptiveMaxPool2d((2, 2)) → Flatten → Linear(1024, 10)
```

Only the encoder weights are clipped and bimodal-regularized. BatchNorm,
pool, and linear head are unconstrained.

## Run (default whitened input)

```bash
python -m applications.ann_bimodal_classifier \
    --seed 1 --epochs 30 --device cuda --lambda-bimodal 0.1
```

## Run on raw RGB

```bash
python -m applications.ann_bimodal_classifier \
    --input-mode cifar10 --seed 1 --epochs 30 --device cuda --lambda-bimodal 0.1
```

Outputs land under `logs/ann_bimodal_classifier/<input_mode>/seed_<N>/lambda_<L>/`.
