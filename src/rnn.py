from typing import Iterator, overload
from .functions import *
from torch.nn import Parameter
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
        self._free_params = {}

    @overload
    def forward(self, *args):
        self.y = torch.tanh(self.W @ self.y + self.bias)
        return self.y

    def forward(self, x):
        self.y = torch.tanh(self.W @ self.y + self.W_in @ x + self.bias)
        return self.y

    def free_param(self, key: str):
        try:
            p = getattr(self, key)  # type: torch.tensor
            p.requires_grad = True
            self._free_params[key] = p
        except AttributeError:
            print(r"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")

    def fix_param(self, key: str):
        try:
            p = getattr(self, key)  # type: torch.tensor
            p.requires_grad = False
            self._free_params.pop(key)
        except AttributeError:
            print(r"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")

    def set_param(self, key: str, val: torch.tensor):
        try:
            setattr(self, key, val)
        except AttributeError:
            print(r"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in self._free_params.values():
            yield p


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


class ConceptorRNN(RNN):

    def __init__(self, N: int, n_in: int, sr: float = 1.0, density: float = 0.2, bias_var: float = 0.2,
                 rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        super().__init__(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        self.conceptors = {}
        self.C = torch.zeros((N, N), device=device, dtype=dtype)

    def learn_conceptor(self, key: str, y: torch.tensor, alpha: float = 10.0):

        R = (y.T @ y) / y.shape[0]
        U, S, V = torch.linalg.svd(R, full_matrices=True)
        S = torch.diag(S)
        S = S @ torch.linalg.inv(S + torch.eye(self.N)/alpha**2)
        self.conceptors[key] = U @ S @ U.T
        self.C = self.conceptors[key]

    def activate_conceptor(self, key: str):
        self.C = self.conceptors[key]

    @overload
    def forward_c(self, *args):
        self.y = self.C @ torch.tanh(self.W @ self.y + self.bias)
        return self.y

    def forward_c(self, x):
        y = super().forward(x)
        self.y = self.C @ y
        return self.y

    def load_pattern(self, x, tychinov_alpha: float = 1e-4) -> float:

        bias = torch.tile(self.bias, (x.shape[0], 1)).T
        targets = np.arctanh(x) - bias
        x_old = torch.zeros_like(x)
        x_old[1:, :] = x[:-1, :]
        self.W = tensor_ridge(x_old, targets, tychinov_alpha)
        return tensor_nrmse(x_old @ self.W.T, targets)


class AutoConceptorRNN(ConceptorRNN):

    def __init__(self, N: int, n_in: int, lam: float = 0.01, alpha: float = 10.0, sr: float = 1.0, density: float = 0.2,
                 bias_var: float = 0.2, rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        super().__init__(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        self.lam = lam
        self.alpha_sq = alpha**(-2)

    @overload
    def forward_c_adapt(self, *args):
        y = self.C @ super().forward()
        self.C = self.C + self.lam * ((self.y - self.C @ self.y) @ self.y.T) - self.C * self.alpha_sq
        self.y = y
        return y

    def forward_c_adapt(self, x):
        y = self.C @ super().forward(x)
        self.C = self.C + self.lam*((self.y - self.C @ self.y) @ self.y.T) - self.C*self.alpha_sq
        self.y = y
        return y

    def store_conceptor(self, key: str):
        self.conceptors[key] = self.C


class RandomFeatureConceptorRNN(LowRankRNN):

    def __init__(self, N: int, n_in: int, rank: int = 1, lam: float = 0.01, alpha: float = 10.0, sr: float = 1.0,
                 density: float = 0.2, bias_var: float = 0.2, rf_var: float = 1.0, device: str = "cpu",
                 dtype: torch.dtype = torch.float64):

        super().__init__(N, n_in, rank, sr, density, bias_var, rf_var, device, dtype)
        self.alpha_sq = alpha**(-2)
        self.lam = lam
        self.C = torch.zeros((rank,), device=device, dtype=dtype)
        self.conceptors = {}
        self.z = torch.zeros_like(self.C)

    @overload
    def forward(self, *args):
        self.y = torch.tanh(self.A @ self.z + self.bias)
        self.z = self.B @ self.y
        return self.y

    def forward(self, x):
        self.y = torch.tanh(self.A @ self.z + self.W_in @ x + self.bias)
        self.z = self.B @ self.y
        return self.y

    @overload
    def forward_c(self, *args):
        self.y = torch.tanh(self.A @ self.z + self.bias)
        self.z = torch.diag(self.C) * self.B @ self.y
        return self.y

    def forward_c(self, x):
        self.y = torch.tanh(self.A @ self.z + self.W_in @ x + self.bias)
        self.z = torch.diag(self.C) * self.B @ self.y
        return self.y

    @overload
    def forward_c_adapt(self, *args):
        self.y = torch.tanh(self.A @ self.z + self.bias)
        z = torch.diag(self.C) * self.B @ self.y
        self.C = self.C + self.lam * (z**2 - self.C*z**2 - self.C*self.alpha_sq)
        self.z = z
        return self.y

    def forward_c_adapt(self, x):
        self.y = torch.tanh(self.A @ self.z + self.W_in @ x + self.bias)
        z = torch.diag(self.C) * self.B @ self.y
        self.C = self.C + self.lam * (z ** 2 - self.C * z ** 2 - self.C * self.alpha_sq)
        self.z = z
        return self.y

    def activate_conceptor(self, key: str):
        self.C = self.conceptors[key]

    def store_conceptor(self, key: str):
        self.conceptors[key] = self.C
