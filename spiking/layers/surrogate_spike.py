import torch
from torch import Tensor
from torch.autograd.function import Function, FunctionCtx


class SurrogateSpike(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx, membrane_potential: Tensor, threshold: Tensor
    ) -> Tensor:
        ctx.save_for_backward(membrane_potential - threshold)
        return (membrane_potential >= threshold).float()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        (delta_u,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_surrogate = 0.5 / (1.0 + torch.abs(delta_u)) ** 2  # fast sigmoid

        grad_u = grad_input * grad_surrogate
        grad_theta = -grad_u

        return grad_u, grad_theta
