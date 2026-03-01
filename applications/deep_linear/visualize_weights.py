import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spiking import SpikingModule


def save_weight_figure(
    model: SpikingModule,
    spike_shape: tuple[int, ...],
    path: str,
    ncols: int = 8,
):
    """Save a grid of per-neuron weight images to a PNG file.

    spike_shape: (2, H, W) — the full spike volume shape.
    Each neuron's weights are reshaped to (2, H, W), padded to (3, H, W) for RGB.
    """
    image_shape = spike_shape[1:]
    num_neurons = model.num_outputs
    nrows = (num_neurons - 1) // ncols + 1

    fig = plt.figure(figsize=(ncols * 2, nrows * 2))
    for idx in range(num_neurons):
        img = model.weights[idx].reshape((2, *image_shape))
        padding = torch.zeros((1, *image_shape), device=img.device)
        img = torch.cat([img, padding], dim=0)
        img = img.permute(1, 2, 0).detach().cpu().numpy()

        plt.subplot(nrows, ncols, idx + 1)
        plt.title(str(idx), fontsize=8)
        plt.axis("off")
        plt.imshow(img)

    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
