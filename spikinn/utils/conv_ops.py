import torch
import torch.nn.functional as F


def unfold_patches(
    input_times: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding: int,
) -> torch.Tensor:
    has_batch = input_times.dim() == 4
    if padding == 0 and input_times.shape[-2:] == (kernel_size, kernel_size):
        if has_batch:
            return input_times.reshape(input_times.shape[0], -1).unsqueeze(1)
        return input_times.flatten().unsqueeze(0)

    if not has_batch:
        input_times = input_times.unsqueeze(0)
    if padding > 0:
        # pad with inf so out-of-bounds taps never contribute a spike
        input_times = F.pad(input_times, [padding] * 4, value=float("inf"))
    patches = F.unfold(input_times, kernel_size=kernel_size, padding=0, stride=stride)
    patches = patches.permute(0, 2, 1)
    return patches if has_batch else patches.squeeze(0)
