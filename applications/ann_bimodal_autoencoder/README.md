# ANN Bimodal Autoencoder

Trains a single-layer convolutional autoencoder on raw CIFAR-10 RGB to test
whether STDP-style bimodal weight distributions (peaks near 0 and 1) emerge
in a plain ANN when the encoder is bounded to [0, 1] and a smooth bimodal
penalty is added to the reconstruction loss. Decoder is unconstrained so the
model retains the capacity to reconstruct.

## Architecture

- Encoder: `Conv2d(3, 256, 5, padding=2, bias=False)` → ReLU; weights init `U(0,1)`,
  hard-clipped to `[0, 1]` after every optimizer step.
- `BatchNorm2d(256)` (downstream of encoder; keeps decoder sigmoid out of saturation).
- Decoder: `ConvTranspose2d(256, 3, 5, padding=2, bias=True)` → Sigmoid; unconstrained.

## Loss

```
L_recon   = sqrt(mean((x_recon - x)^2))                # RMSE
L_bimodal = mean( min(w**2, (w-1)**2) ) over encoder w
L_total   = L_recon + λ * L_bimodal
```

## Run a single configuration

```bash
python -m applications.ann_bimodal_autoencoder \
    --seed 1 --epochs 30 --batch-size 128 --lr 1e-3 \
    --lambda-bimodal 0.1 \
    --device cuda
```

## Run the λ sweep

```bash
python -m applications.ann_bimodal_autoencoder \
    --seed 1 --epochs 30 --device cuda \
    --lambda-bimodal-sweep "0.0,0.01,0.1,1.0"
```

Outputs land in `logs/ann_bimodal_autoencoder/<dataset>/seed_<N>/lambda_<L>/`
(or `.../seed_<N>/sweep/lambda_<L>/` for sweeps): weight histograms
(init / mid / final), filter grids, reconstructions, loss curves,
bimodality-over-time, and (for sweeps) `lambda_sweep.png`.

## Smoke test

```bash
python -m applications.ann_bimodal_autoencoder \
    --epochs 2 --train-subset 2000 --lambda-bimodal 0.1 \
    --output-dir logs/ann_bimodal_autoencoder/_smoke/quick
```
