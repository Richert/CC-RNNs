from .functions import *
import numpy as np


class Reservoir:
    
    def __init__(self, N: int, n_in: int, alpha: float = 10.0, sr: float = 1.5, bias_scale: float = 0.2,
                 inp_scale: float = 1.5, density: float = 0.1):
        """
        Initializes the reservoir with the provided parameters.

        :param N: the number of neurons you would like to have in the reservoir.
        :param n_in: The number of inputs to the reservoir
        :param alpha: The aperture control parameter.
        :param sr: spectral radius
        :param bias_scale: scaling of the bias term to be introduced to each neuron.
        :param inp_scale: the scale for the input weights.
        :param density: relative fraction of reservoir connections established in the reservoir.
        """
                 
        self.N = N
        self.alpha = alpha
        self.W = sr * IntWeights(self.N, self.N, density)
        self.W_bias = bias_scale*np.random.randn(self.N)
        self.W_in = inp_scale*np.random.randn(self.N, n_in)
        self.W_out = np.zeros((n_in, self.N))
        self.n_patterns = 0
        self.readout_error = 0.0
        self.loading_error = 0.0
        self.C = []
        self.y = np.zeros((N,), dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.dot(self.W, self.y) + np.dot(self.W_in, x) + self.W_bias)

    def forward_auto(self) -> np.ndarray:
        return np.tanh(np.dot(self.W, self.y) + self.W_bias)

    def run(self, patterns, learning_steps: int = 1000, init_steps: int = 100, c_adapt_rate: float = None,
            load_reservoir: bool = True, tychinov_alphas: tuple = (0.01, 0.001), **kwargs):

        # process user input
        self.n_patterns = len(patterns)
        n_in = self.W_in.shape[1]
        I = np.eye(self.N)
        gradient_cut = kwargs.pop("gradient_cut", 2.0)
        sgd = False if c_adapt_rate is None else True
        
        # initialize storage matrices
        states = np.zeros([self.N, self.n_patterns * learning_steps])
        targets = np.zeros([n_in, self.n_patterns * learning_steps])
        
        # train conceptors     
        for i, p in enumerate(patterns):

            y_col = np.zeros([self.N, learning_steps])
            x_col = np.zeros([n_in, learning_steps])
            Cc = np.zeros([self.N, self.N])            

            for step in range(learning_steps+init_steps):
                
                if not type(p == np.ndarray):
                    x = np.reshape(p(step), n_in)
                else:
                    x = p[step]

                # calculate new state
                y = self.forward(x)

                if sgd:

                    # SGD conceptor update
                    grad = x - Cc @ x
                    norm = np.linalg.norm(grad)     
                    if norm > gradient_cut:
                        grad = gradient_cut/norm * grad
                    Cc = Cc + c_adapt_rate*(np.outer(grad, x.T) - (self.alpha**-2)*Cc)
                
                if step >= init_steps:

                    # state collection
                    y_col[:, step-init_steps] = y
                    x_col[:, step-init_steps] = x
            
            if sgd:

                # store SGD-trained conceptor
                self.C.append(Cc)

            else:

                # train conceptor on collected states
                try:
                    R = np.dot(y_col,np.transpose(y_col)) / learning_steps
                    U, S, V = np.linalg.svd(R, full_matrices=True)
                    S = np.diag(S)
                    S = (np.dot(S, np.linalg.inv(S + (self.alpha**-2)*I)))
                    self.C.append(np.dot(U, np.dot(S, U.T)))
                except ValueError:
                    print("SVD did not converge")

            states[:, i*learning_steps:(i+1)*learning_steps] = y_col
            targets[:, i*learning_steps:(i+1)*learning_steps] = x_col
        
        if load_reservoir:
        
            # train the readout weights and calculate the readout error
            self.W_out = ridge(states, targets, tychinov_alphas[0])
            self.readout_error = nrmse(np.dot(self.W_out, states), targets)
            print(self.readout_error)
            
            # load the
            W_bias_rep = np.tile(self.W_bias,(self.n_patterns*learning_steps,1)).T
            W_targets = np.arctanh(states) - W_bias_rep
            states_old = np.zeros_like(states)
            states_old[:, 1:] = states[:, :-1]
            self.W = ridge(states_old, W_targets, tychinov_alphas[1])
            self.loading_error = nrmse(np.dot(self.W, states_old), W_targets)
            print(np.mean(self.loading_error))
        
        states = np.reshape(states, [self.n_patterns, self.N, learning_steps])
        targets = np.reshape(targets, [self.n_patterns, n_in, learning_steps])

        return states, targets

    def recall(self, recall_steps: int = 200, init_noise: float = 0.5) -> np.ndarray:
        
        z_col = []
        
        for i in range(self.n_patterns):
            
            Cc = self.C[i]
            self.y = init_noise * np.random.randn(self.N)
            z_recall = np.zeros([recall_steps, self.W_in.shape[1]])
        
            for step in range(recall_steps):
                
                self.y = np.dot(Cc, self.forward_auto())
                z = np.dot(self.W_out, self.y)
                z_recall[step] = z
                
            z_col.append(z_recall)

        return np.asarray(z_col)
