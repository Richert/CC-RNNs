import numpy as np
from functions import *
import torch


class RNN(torch.nn.Module):

    def __init__(self, N: int, n_in: int, sr: float = 1.0, density: float = 0.2, bias_var: float = 0.2,
                 rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        super().__init__()
        self.N = N
        self.W = torch.tensor(sr * init_weights(self.N, self.N, density), device=device, dtype=dtype)
        self.W_in = torch.tensor(rf_var*np.random.randn(self.N, n_in), device=device, dtype=dtype)
        self.bias = torch.tensor(bias_var*np.random.randn(self.N), device=device, dtype=dtype)
        self.y = torch.zeros((self.N,), device=device, dtype=dtype)

    def forward(self, x):
        self.y = torch.tanh(self.W @ self.y + self.W_in @ x + self.bias)
        return self.y

    def free_param(self, key: str):
        try:
            p = getattr(self, key)  # type: torch.tensor
            p.requires_grad = True
        except AttributeError:
            print(r"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")

    def hold_param(self, key: str):
        try:
            p = getattr(self, key)  # type: torch.tensor
            p.requires_grad = False
        except AttributeError:
            print(r"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")


class LowRankRNN(RNN):

    def __init__(self, N: int, n_in: int, rank: int = 1, sr: float = 1.0, density: float = 0.2, bias_var: float = 0.2,
                 rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        super().__init__(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        self.A = torch.tensor(sr * init_weights(self.N, rank, density), device=device, dtype=dtype)
        self.B = torch.tensor(sr * init_weights(rank, self.N, density), device=device, dtype=dtype)
        self.W = [self.A, self.B]

    def forward(self, x):
        self.y = torch.tanh(self.A @ self.B @ self.y + self.W_in @ x + self.bias)
        return self.y
