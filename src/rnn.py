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

    def forward(self, x):
        self.y = torch.tanh(self.W @ self.y + self.W_in @ x + self.bias)
        return self.y

    def forward_a(self, D):
        self.y = torch.tanh((self.W + D) @ self.y + self.bias)
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

    def load_input(self, u, x, tychinov_alpha: float = 1e-4) -> tuple:
        assert u.shape[-1] == x.shape[-1]  # time steps
        targets = self.W_in @ u
        D = tensor_ridge(x, targets, tychinov_alpha)
        return D, tensor_nrmse(D @ x, targets)


class LowRankRNN(RNN):

    def __init__(self, N: int, n_in: int, rank: int = 1, sr: float = 1.0, density: float = 0.2, bias_var: float = 0.2,
                 rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        super().__init__(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        A = init_weights(N, rank, density)
        B = init_weights(rank, N, density)
        sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(A, B))))
        A *= np.sqrt(sr) / np.sqrt(sr_comb)
        B *= np.sqrt(sr) / np.sqrt(sr_comb)
        self.A = torch.tensor(A, device=device, dtype=dtype)
        self.B = torch.tensor(B, device=device, dtype=dtype)
        self.W = [self.A, self.B]
        self.z = torch.zeros((rank,), device=device, dtype=dtype)

    def forward(self, x):
        self.y = torch.tanh(self.A @ self.z + self.W_in @ x + self.bias)
        self.z = self.B @ self.y
        return self.y

    def forward_a(self, D):
        self.y = torch.tanh(self.A @ self.z + D @ self.y + self.bias)
        self.z = self.B @ self.y
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

    def forward_c(self, x):
        self.y = self.C @ super().forward(x)
        return self.y

    def forward_c_a(self, D):
        self.y = self.C @ super().forward_a(D)
        return self.y


class AutoConceptorRNN(ConceptorRNN):

    def __init__(self, N: int, n_in: int, lam: float = 0.01, alpha: float = 10.0, sr: float = 1.0, density: float = 0.2,
                 bias_var: float = 0.2, rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        super().__init__(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        self.lam = lam
        self.alpha_sq = alpha**(-2)

    def forward_c_a_adapt(self, D):
        y = self.C @ torch.tanh((self.W + D) @ self.y + self.bias)
        self.C = self.C + self.lam * ((self.y - self.C @ self.y) @ self.y.T) - self.C * self.alpha_sq
        self.y = y
        return y

    def forward_c_adapt(self, x):
        y = self.C @ torch.tanh(self.W @ self.y + self.W_in @ x + self.bias)
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
        self.C = torch.ones_like(self.z)
        self.conceptors = {}

    def forward_c(self, x):
        self.y = torch.tanh(self.A @ self.z + self.W_in @ x + self.bias)
        self.z = self.C * (self.B @ self.y)
        return self.y

    def forward_c_a(self, D):
        self.y = torch.tanh(self.A @ self.z + D @ self.y + self.bias)
        self.z = self.C * (self.B @ self.y)
        return self.y

    def forward_c_adapt(self, x):
        self.y = torch.tanh(self.A @ self.z + self.W_in @ x + self.bias)
        z = self.C * (self.B @ self.y)
        self.C = self.C + self.lam * (self.z ** 2 - self.C * self.z ** 2 - self.C * self.alpha_sq)
        self.z = z
        return self.y

    def forward_c_a_adapt(self, D):
        self.y = torch.tanh(self.A @ self.z + D @ self.y + self.bias)
        z = self.C * (self.B @ self.y)
        self.C = self.C + self.lam * (self.z**2 - self.C*self.z**2 - self.C*self.alpha_sq)
        self.z = z
        return self.y

    def activate_conceptor(self, key: str):
        self.C = self.conceptors[key]

    def store_conceptor(self, key: str):
        self.conceptors[key] = self.C
