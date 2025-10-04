import torch
from torch.autograd.function import Function, FunctionCtx


def custom_fast_sigmoid_grad(delta_u, grad_input, slope=25.0):
    grad_surrogate = 1.0 / (slope * delta_u.abs() + 1.0) ** 2
    return grad_input * grad_surrogate


class SurrogateSpike(Function):
    @classmethod
    def forward(
        cls, ctx: FunctionCtx, membrane_potential: torch.Tensor, threshold: torch.Tensor
    ):
        ctx.save_for_backward(membrane_potential - threshold)
        return (membrane_potential >= threshold).float()

    @classmethod
    def backward(cls, ctx: FunctionCtx, grad_output: torch.Tensor):
        (delta_u,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = custom_fast_sigmoid_grad(delta_u, grad_input)
        return grad, -grad
