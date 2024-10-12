import random

import numpy
import torch


def setup_reproducibility() -> None:
	random.seed(37710)  # Sets the seed for Python's built-in random module
	numpy.random.seed(37710)  # Sets the seed for NumPy's random number generator

	torch.manual_seed(37710)  # Sets the seed for PyTorch's CPU random number generator
	torch.cuda.manual_seed(37710)  # Sets the seed for the current GPU device
	torch.cuda.manual_seed_all(37710)  # Sets the seed for all available GPU devices

	torch.use_deterministic_algorithms(True)  # Ensures that only deterministic algorithms are used


def get_torch_device() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"

	return "cpu"
