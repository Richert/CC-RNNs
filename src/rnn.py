from typing import Iterator, Iterable, Union, Callable
from .functions import *
from torch.nn import Parameter
import torch


class RNN(torch.nn.Module):

    def __init__(self, W: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor, f: Union[str, Callable] = "Tanh"):

        super().__init__()
        self.N = W.shape[0]
        self.device = W.device
        self.dtype = W.dtype
        self.W = W
        self.W_in = W_in
        self.D = torch.zeros_like(W)
        self.bias = bias
        self.y = torch.zeros((self.N,), device=self.device, dtype=self.dtype)
        self.f = self._get_activation_function(f)
        self._free_params = {}
        self._state_vars = ["y"]

    @property
    def state_vars(self):
        for v in self._state_vars:
            yield getattr(self, v)

    def forward(self, x):
        self._update_layer(self._inp_layer(x))
        return self._out_layer()

    def forward_a(self):
        self._update_layer(self._noinp_layer())
        return self._out_layer()

    def get_vf(self, L: torch.Tensor, R: torch.Tensor, grid_points: int, lower_bounds: Iterable, upper_bounds: Iterable,
               *args, **kwargs) -> tuple:

        # get coordinates at which to evaluate the vectorfield
        eval_points = [np.linspace(lb, ub, grid_points) for lb, ub in zip(lower_bounds, upper_bounds)]
        coordinates = np.asarray([c.flatten(order="C") for c in np.meshgrid(*tuple(eval_points), **kwargs)])
        coordinates = torch.tensor(coordinates, device=self.y.device, dtype=self.y.dtype).T

        # calculate coordinate inversion matrix
        R_inv = torch.linalg.lstsq(R, torch.eye(R.shape[0], dtype=R.dtype, device=R.device)).solution

        # evaluate the vectorfield
        vf = torch.zeros_like(coordinates)
        with torch.no_grad():
            for idx in range(coordinates.shape[0]):
                z = coordinates[idx, :]
                y = R_inv @ z
                self.set_state(y, z)
                y_new = self.forward_a()
                vf[idx, :] = R @ (y_new - y)

        return coordinates, vf

    def get_io_transform(self, x_min: float, x_max: float, n: int = 1000):
        x = torch.linspace(x_min, x_max, steps=n)
        return x, self.f(x)

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
        for var in self._state_vars:
            val = getattr(self, var)
            setattr(self, var, val.detach())

    def set_state(self, values):
        for var, val in zip(self._state_vars, values):
            setattr(self, var, val)

    def clip(self, val: float):
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -val, val))
        for var in self.state_vars:
            if var.requires_grad:
                var.register_hook(lambda grad: torch.clamp(grad, -val, val))

    @staticmethod
    def train_readout(y, target, tychinov_alpha: float = 1e-4) -> tuple:
        assert target.shape[-1] == y.shape[-1]  # time steps
        W_readout = tensor_ridge(y, target, tychinov_alpha)
        return W_readout, tensor_nrmse(W_readout @ y, target)

    def _inp_layer(self, x):
        return self.W_in @ x

    def _noinp_layer(self):
        return self.D @ self.y + self.bias

    def _out_layer(self):
        return self.y

    def _update_layer(self, x):
        self.y = self.f(self.W @ self.y + x + self.bias)

    @staticmethod
    def _get_activation_function(f: Union[str, Callable]):
        if type(f) is Callable:
            return f
        try:
            f = getattr(torch.nn, f)
            return f()
        except AttributeError as e:
            raise e


class LowRankRNN(RNN):

    def __init__(self, W: torch.Tensor, L: torch.Tensor, R: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor,
                 f: Union[str, Callable] = "Tanh", g: Union[str, Callable] = "Identity"):

        super().__init__(W, W_in, bias, f=f)
        self.L = L
        self.R = R
        self.rank = L.shape[1]
        self.z = torch.zeros((self.rank,), device=self.device, dtype=self.dtype)
        self.g = self._get_activation_function(g)
        self._state_vars = ["y", "z"]

    def get_vf(self, grid_points: int, lower_bounds: Iterable, upper_bounds: Iterable, *args, **kwargs) -> tuple:
        return super().get_vf(self.L, self.R, grid_points, lower_bounds, upper_bounds, *args, **kwargs)

    def _out_layer(self):
        return self.z

    def _update_layer(self, x):
        self.y = self.f(self.W @ self.y + self.L @ self.z + x)
        self.z = self.g(self.R @ self.y + self.bias)


class LowRankCRNN(LowRankRNN):

    def __init__(self, W: torch.Tensor, L: torch.Tensor, R: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor,
                 lam: float = 1e-3, alpha: float = 1.0, g: Union[str, Callable] = "Identity"):

        super().__init__(W, L, R, W_in, bias, g=g)
        self.alpha_sq = alpha ** (-2)
        self.lam = lam
        self.z_controllers = {}
        self.y_controllers = {}
        self.C_z = torch.ones_like(self.z, device=self.device)
        self.C_y = torch.ones_like(self.y, device=self.device)

    def update_z_controller(self):
        self.C_z = self.C_z + self.lam * (self.z ** 2 - self.C_z * self.z ** 2 - self.C_z * self.alpha_sq)

    def update_y_controller(self):
        self.C_y = self.C_y + self.lam * (self.y ** 2 - self.C_y * self.y ** 2 - self.C_y * self.alpha_sq)

    def activate_z_controller(self, key):
        self.C_z = self.z_controllers[key]

    def activate_y_controller(self, key):
        self.C_y = self.y_controllers[key]

    def store_z_controller(self, key):
        self.z_controllers[key] = self.C_z

    def store_y_controller(self, key):
        self.y_controllers[key] = self.C_y

    def init_new_z_controller(self, init_value: str = "zero"):
        if init_value == "zero":
            self.C_z = torch.zeros_like(self.z)
        elif init_value == "random":
            self.C_z = torch.rand(size=self.z.size(), dtype=self.z.dtype, device=self.z.device)
        else:
            self.C_z = torch.ones_like(self.z, device=self.device)

    def init_new_y_controller(self, init_value: Union[str, torch.Tensor] = "zero"):
        if init_value == "zero":
            self.C_y = torch.zeros_like(self.y)
        elif init_value == "random":
            self.C_y = torch.rand(size=self.y.size(), dtype=self.y.dtype, device=self.y.device)
        elif init_value == "ones":
            self.C_y = torch.ones_like(self.y, device=self.device)
        else:
            self.C_y = init_value.to(self.device)

    def detach(self):
        super().detach()
        self.C_z = self.C_z.detach()
        self.C_y = self.C_y.detach()

    def _update_layer(self, x):
        self.y = self.C_y * self.f(self.W @ self.y + self.L @ self.z + x)
        self.z = self.C_z * self.g(self.R @ self.y + self.bias)

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


class HHRNN(LowRankCRNN):

    def __init__(self, L: torch.Tensor, R: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor,
                 g_s: float = 120.0, g_p: float = 36.0, g_l: float = 0.3, E_s: float = 50.0, E_p: float = -77.0,
                 E_l: float = -54.4, C: float = 1.0, gamma: float = 1.0, theta: float = 30.0,
                 dt: float = 1e-3, g: Union[str, Callable] = "Identity", alpha: float = 100.0, lam: float = 1e-3):

        super().__init__(L, R, W_in, bias, g=g, alpha=alpha, lam=lam)
        self.g_s = g_s
        self.g_p = g_p
        self.g_l = g_l
        self.E_s = E_s
        self.E_p = E_p
        self.E_l = E_l
        self.C = C
        self.u = torch.zeros_like(self.y)
        self.w = torch.zeros_like(self.z)
        self.dt = dt
        self.gamma = gamma
        self.theta = theta
        self._state_vars = ["y", "u", "z", "w"]

    def f(self, x):
        return torch.sigmoid(self.gamma*(x-self.theta))

    def _out_layer(self):
        return self.f(self.z)

    def _update_layer(self, x):
        dy, du, dz, dw = self._rhs_layer(x, self.y, self.u, self.z, self.w)
        self.y = self.y + self.dt * dy
        self.u = self.u + self.dt * du
        self.z = self.z + self.dt * dz
        self.w = self.w + self.dt * dw

    def _heun_update_layer(self, x):
        dy, du, dz, dw = self._rhs_layer(x, self.y, self.u, self.z, self.w)
        y_tmp = self.y + 0.5*self.dt * dy
        u_tmp = self.u + 0.5*self.dt * du
        z_tmp = self.z + 0.5*self.dt * dz
        w_tmp = self.w + 0.5*self.dt * dw
        dy, du, dz, dw = self._rhs_layer(x, y_tmp, u_tmp, z_tmp, w_tmp)
        self.y = self.y + self.dt * dy
        self.u = self.u + self.dt * du
        self.z = self.z + self.dt * dz
        self.w = self.w + self.dt * dw

    def _rhs_layer(self, x, y, u, z, w):
        dy = (self.W @ self.f(z) + x - self._sodium(y, u) - self._potassium(y, u) - self._leak(y)) / self.C
        du = self._alpha_n(y) * (1 - u) - self._beta_n(y) * u
        dz = (self.g(self.W_z @ y - z) - self._sodium(z, w) - self._potassium(z, w) - self._leak(z)) / self.C
        dw = self._alpha_n(z) * (1 - w) - self._beta_n(z) * w
        return dy, du, dz, dw

    def _sodium(self, v, n):
        alpha = self._alpha_m(v)
        return self.g_s*(v-self.E_s)*(0.8-n)*(alpha/(alpha + self._beta_m(v)))**3

    def _potassium(self, v, n):
        return self.g_p*(v-self.E_p)*n**4

    def _leak(self, v):
        return self.g_l*(v-self.E_l)

    @staticmethod
    def _alpha_n(v: torch.Tensor):
        return 0.01*(v + 55.0)/(1.0 - torch.exp(-(v+55.0)/10.0))

    @staticmethod
    def _beta_n(v: torch.Tensor):
        return 0.125*torch.exp(-(v+65.0)/80.0)

    @staticmethod
    def _alpha_m(v: torch.Tensor):
        return 0.1*(v+40.0)/(1.0 - torch.exp(-(v+40.0)/10.0))

    @staticmethod
    def _beta_m(v: torch.Tensor):
        return 4.0*torch.exp(-(v+65.0)/18.0)


class FHNRNN(LowRankRNN):

    def __init__(self, L: torch.Tensor, R: torch.Tensor, W_in: torch.Tensor, bias: torch.Tensor,
                 k: float = 1.0, a: float = 0.7, b: float = 0.8, tau: float = 12.5, gamma: float = 100.0, theta: float = 1.5,
                 dt: float = 1e-3, g: Union[str, Callable] = "Identity"):

        super().__init__(L, R, W_in, bias, g=g)
        self.k = k
        self.a = a
        self.b = b
        self.tau = tau
        self.gamma = gamma
        self.theta = theta
        self.u = torch.zeros_like(self.y)
        self.w = torch.zeros_like(self.z)
        self.dt = dt
        self._state_vars = ["y", "u", "z", "w"]

    def f(self, x):
        return 1.0/(1.0 + torch.exp(-self.gamma*(x-self.theta)))

    def _out_layer(self):
        return self.f(self.z)

    def _update_layer(self, x):
        dy = self.k*(self.y - self.y**3/3) - self.u + self.W @ self.f(self.z) + x
        du = (self.y + self.a - self.b * self.u) / self.tau
        dz = self.k*(self.z - self.z**3/3) - self.w + self.W_z @ self.g(self.y)
        dw = (self.z + self.a - self.b * self.w) / self.tau
        self.y = self.y + self.dt * dy
        self.u = self.u + self.dt * du
        self.z = self.z + self.dt * dz
        self.w = self.w + self.dt * dw


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

    def _rhs_layer(self, x):
        return self.C @ super()._rhs_layer(x)


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

    def store_conceptor(self, key: str):
        self.conceptors[key] = self.C

    def _update_layer(self, y):
        super()._update_layer(y)
        self.C = self.C + self.lam * ((self.y - self.C @ self.y) @ self.y.T) - self.C * self.alpha_sq
