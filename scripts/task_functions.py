import torch
import numpy as np


def cycling(freq: float, trial_dur: int, min_cycling_dur: int, inp_dur: int, inp_damping: float, trials: int,
            noise: float, dt: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> tuple:

    # create inputs and targets
    inputs = np.zeros((trials, trial_dur, 2)) + noise * np.random.randn(trials, trial_dur, 2)
    targets = np.zeros((trials, trial_dur, 1))
    for trial in range(trials):
        start = np.random.randint(low=0, high=int(0.5*(trial_dur-min_cycling_dur)))
        stop = np.random.randint(low=start+min_cycling_dur, high=trial_dur - inp_dur)
        inputs[trial, start:start+inp_dur, 0] += 1.0
        inputs[trial, stop:stop+inp_dur, 1] += 1.0
        sine = np.sin(2.0*np.pi*freq*np.linspace(0.0, (stop-start)*dt, stop-start))
        damping = (np.ones((stop-start,))*inp_damping)**np.arange(1, stop-start+1)
        targets[trial, start:stop, 0] = 2.0 * sine * damping

    return torch.tensor(inputs, device=device, dtype=dtype), torch.tensor(targets, device=device, dtype=dtype)


def delayed_choice(trials: int, evidence: int, delay_min: int, delay_max: int, response: int, noise: float) -> tuple:

    # allocate arrays
    inputs, targets = [], []

    # create inputs and targets
    for trial in range(trials):

        # choose random delay
        delay = np.random.randint(low=delay_min, high=delay_max)
        trial_dur = evidence + delay + response

        # initialize arrays
        trial_inp = np.random.randn(trial_dur, 3) * noise
        trial_targ = np.zeros((trial_dur, 1))

        # choose random input channel
        channel = np.random.randint(low=0, high=2)
        trial_inp[:evidence, channel] += 1.0
        trial_inp[evidence+delay:, 2] += 1.0
        trial_targ[evidence+delay:, 0] = 5.0 if channel == 0 else -5.0

        # save trial inputs and targets
        inputs.append(trial_inp)
        targets.append(trial_targ)

    return inputs, targets


def init_state(x: torch.Tensor, noise: float, boundaries: tuple, device: str = "cpu",
               dtype: torch.dtype = torch.float64) -> torch.Tensor:
    x = x + noise * torch.randn(x.shape, device=device, dtype=dtype)
    if boundaries:
        x[x < boundaries[0]] = boundaries[0]
        x[x > boundaries[1]] = boundaries[1]
    return x


