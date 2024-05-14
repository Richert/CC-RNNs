from typing import Iterator, overload
from .functions import *
from torch.nn import Parameter
import torch


class RNN(torch.nn.Module):

    def __init__(self, W: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor):

        super().__init__()
        self.N = W.shape[0]
        self.device = W.device
        self.dtype = W.dtype
        self.W = W
        self.W_in = W_in
        self.D = None
        self.bias = bias
        self.y = torch.zeros((self.N,), device=self.device, dtype=self.dtype)
        self._free_params = {}

    @classmethod
    def random_init(cls, N: int, n_in: int, sr: float = 1.0, density: float = 0.2, bias_var: float = 0.2,
                    rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        W = torch.tensor(sr * init_weights(N, N, density), device=device, dtype=dtype)
        W_in = torch.tensor(rf_var * np.random.randn(N, n_in), device=device, dtype=dtype)
        bias = torch.tensor(bias_var * np.random.randn(N), device=device, dtype=dtype)
        return cls(W, W_in, bias)

    def forward(self, x):
        self.y = torch.tanh(self.W @ self.y + self.W_in @ x + self.bias)
        return self.y

    def forward_a(self):
        self.y = torch.tanh((self.W + self.D) @ self.y + self.bias)
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
            try:
                self._free_params.pop(key)
            except KeyError:
                pass
        except AttributeError:
            print(f"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")

    def set_param(self, key: str, val: torch.tensor):
        try:
            setattr(self, key, val)
        except AttributeError:
            print(r"Invalid parameter key {key}. Key should refer to a tensor object available as an attribute on RNN.")

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in self._free_params.values():
            yield p

    def load_input(self, y, target, tychinov_alpha: float = 1e-4, overwrite: bool = False) -> tuple:
        D, epsilon = self.train_readout(y, target, tychinov_alpha=tychinov_alpha)
        if overwrite or self.D is None:
            self.D = self.W_in @ D
        else:
            self.D += self.W_in @ D
        return D, epsilon

    def detach(self):
        self.y = self.y.detach()

    @staticmethod
    def train_readout(y, target, tychinov_alpha: float = 1e-4) -> tuple:
        assert target.shape[-1] == y.shape[-1]  # time steps
        W_readout = tensor_ridge(y, target, tychinov_alpha)
        return W_readout, tensor_nrmse(W_readout @ y, target)


class LowRankRNN(RNN):

    def __init__(self, W: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor, W_z: torch.Tensor):

        super().__init__(W, W_in, bias)
        self.W_z = W_z
        self.rank = W_z.shape[0]
        self.z = torch.zeros((self.rank,), device=self.device, dtype=self.dtype)

    @classmethod
    def random_init(cls, N: int, n_in: int, rank: int = 1, sr: float = 1.0, density: float = 0.2, bias_var: float = 0.2,
                    rf_var: float = 1.0, device: str = "cpu", dtype: torch.dtype = torch.float64):

        rnn = super().random_init(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        W = init_weights(N, rank, density)
        W_z = init_weights(rank, N, density)
        sr_comb = np.max(np.abs(np.linalg.eigvals(np.dot(W, W_z))))
        W *= np.sqrt(sr) / np.sqrt(sr_comb)
        W_z *= np.sqrt(sr) / np.sqrt(sr_comb)
        return cls(W, rnn.W_in, rnn.bias, W_z)

    def forward(self, x):
        self.y = torch.tanh(self.W @ self.z + self.W_in @ x + self.bias)
        self.z = self.W_z @ self.y
        return self.y

    def forward_a(self):
        self.y = torch.tanh(self.W @ self.z + self.D @ self.y + self.bias)
        self.z = self.W_z @ self.y
        return self.y

    def detach(self):
        super().detach()
        self.z = self.z.detach()


class ConceptorRNN(RNN):

    def __init__(self, W: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor):
        super().__init__(W, W_in, bias)
        self.conceptors = {}
        self.C = torch.zeros((self.N, self.N), device=self.device, dtype=self.dtype)

    def learn_conceptor(self, key, y: torch.tensor, alpha: float = 10.0):

        R = (y.T @ y) / y.shape[0]
        U, S, V = torch.linalg.svd(R, full_matrices=True)
        S = torch.diag(S)
        S = S @ torch.linalg.inv(S + torch.eye(self.N)/alpha**2)
        self.conceptors[key] = U @ S @ U.T
        self.C = self.conceptors[key]
        return self.C

    def activate_conceptor(self, key):
        self.C = self.conceptors[key]

    def forward_c(self, x):
        self.y = self.C @ super().forward(x)
        return self.y

    def forward_c_a(self):
        self.y = self.C @ super().forward_a()
        return self.y


class AutoConceptorRNN(ConceptorRNN):

    def __init__(self, W: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor, lam: float, alpha: float):

        super().__init__(W, W_in, bias)
        self.lam = lam
        self.alpha_sq = alpha**(-2)

    @classmethod
    def random_init(cls, N: int, n_in: int, lam: float = 0.01, alpha: float = 10.0, sr: float = 1.0,
                    density: float = 0.2, bias_var: float = 0.2, rf_var: float = 1.0, device: str = "cpu",
                    dtype: torch.dtype = torch.float64):
        ac = super().random_init(N, n_in, sr, density, bias_var, rf_var, device, dtype)
        return cls(ac.W, ac.W_in, ac.bias, lam, alpha)

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

    def __init__(self, W: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor, W_z: torch.Tensor,
                 lam: float, alpha: float):
        super().__init__(W, W_in, bias, W_z)
        self.alpha_sq = alpha ** (-2)
        self.lam = lam
        self.C = torch.zeros_like(self.z)
        self.conceptors = {}

    @classmethod
    def random_init(cls, N: int, n_in: int, rank: int = 1, lam: float = 0.01, alpha: float = 10.0, sr: float = 1.0,
                    density: float = 0.2, bias_var: float = 0.2, rf_var: float = 1.0, device: str = "cpu",
                    dtype: torch.dtype = torch.float64):
        lr = super().random_init(N, n_in, rank, sr, density, bias_var, rf_var, device, dtype)
        return cls(lr.W, lr.W_in, lr.bias, lr.W_z, lam, alpha)

    def forward_c(self, x):
        self.y = torch.tanh(self.W @ self.z + self.W_in @ x + self.bias)
        self.z = self.C * (self.W_z @ self.y)
        return self.y

    def forward_c_a(self):
        self.y = torch.tanh(self.W @ self.z + self.D @ self.y + self.bias)
        self.z = self.C * (self.W_z @ self.y)
        return self.y

    def forward_c_adapt(self, x):
        self.y = torch.tanh(self.W @ self.z + self.W_in @ x + self.bias)
        z = self.C * (self.W_z @ self.y)
        self.C = self.C + self.lam * (self.z ** 2 - self.C * self.z ** 2 - self.C * self.alpha_sq)
        self.z = z
        return self.y

    def forward_c_a_adapt(self):
        self.y = torch.tanh(self.W @ self.z + self.D @ self.y + self.bias)
        z = self.C * (self.W_z @ self.y)
        self.C = self.C + self.lam * (self.z**2 - self.C*self.z**2 - self.C*self.alpha_sq)
        self.z = z
        return self.y

    def activate_conceptor(self, key):
        self.C = self.conceptors[key]

    def store_conceptor(self, key):
        self.conceptors[key] = self.C

    def init_new_conceptor(self, init_value: str = "zero"):
        if init_value == "zero":
            self.C = torch.zeros_like(self.z)
        elif init_value == "random":
            self.C = torch.rand(size=self.z.size(), dtype=self.z.dtype, device=self.z.device)
        else:
            self.C = torch.ones_like(self.z)

    def detach(self):
        super().detach()
        self.C = self.C.detach()

    @staticmethod
    def combine_conceptors(C1: torch.Tensor, C2: torch.Tensor, operation: str, eps: float = 1e-3) -> torch.Tensor:

        C_comb = torch.zeros_like(C1)
        zero = eps
        one = 1 - eps

        if operation == "and":

            idx = (C1 < zero) * (C2 < zero)
            C_comb[idx == 1.0] = 0.0
            C1_tmp = C1[idx < 1.0]
            C2_tmp = C2[idx < 1.0]
            C_comb[idx < 1.0] = (C1_tmp * C2_tmp) / (C1_tmp + C2_tmp - C1_tmp * C2_tmp)

        elif operation == "or":

            idx = (C1 > one) * (C2 > one)
            C_comb[idx == 1.0] = 1.0
            C1_tmp = C1[idx < 1.0]
            C2_tmp = C2[idx < 1.0]
            C_comb[idx < 1.0] = (C1_tmp + C2_tmp - 2 * C1_tmp * C2_tmp) / (1 - C1_tmp * C2_tmp)

        else:

            raise ValueError(f"Invalid operation for combining conceptors: {operation}.")

        return C_comb
