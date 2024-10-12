import torch


class HMM(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input: torch.Tensor) -> torch.Tensor:
		ctx.save_for_backward(input)
		return 0.5 * (5 * input**3 - 3 * input)

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
		(input,) = ctx.saved_tensors
		return grad_output * 1.5 * (5 * input**2 - 1)
