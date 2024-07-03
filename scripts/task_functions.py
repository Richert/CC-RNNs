import torch
import numpy as np
from typing import Iterable
from itertools import permutations


def cycling(amp: float, freq: float, trial_dur: int, min_cycling_dur: int, inp_dur: int, inp_damping: float, trials: int,
            noise: float, dt: float, device: str = "cpu", dtype: torch.dtype = torch.float64) -> tuple:

    # create inputs and targets
    inputs = np.zeros((trials, trial_dur, 2)) + noise * np.random.randn(trials, trial_dur, 2)
    targets = np.zeros((trials, trial_dur, 1))
    for trial in range(trials):
        start = np.random.randint(low=0, high=int(0.5*(trial_dur-min_cycling_dur)))
        stop = np.random.randint(low=start+min_cycling_dur, high=trial_dur - inp_dur)
        inputs[trial, start:, 0] += 1.0
        inputs[trial, stop:stop+inp_dur, 1] -= 1.0
        sine = np.sin(2.0*np.pi*freq*np.linspace(0.0, (stop-start)*dt, stop-start))
        damping = (np.ones((stop-start,))*inp_damping)**np.arange(1, stop-start+1)
        targets[trial, start:stop, 0] = amp * sine * damping

    return torch.tensor(inputs, device=device, dtype=dtype), torch.tensor(targets, device=device, dtype=dtype)


def delayed_choice(target: float, trials: int, evidence: int, delay_min: int, delay_max: int, response: int, noise: float) -> tuple:

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
        trial_targ[evidence+delay:, 0] = target if channel == 0 else -target

        # save trial inputs and targets
        inputs.append(trial_inp)
        targets.append(trial_targ)

    return inputs, targets


def frequency_choice(amp: float, f1: float, f2: float, trials: int, evidence: int, delay_min: int, delay_max: int,
                     response: int, dt: float, noise: float) -> tuple:

    # allocate arrays
    inputs, targets = [], []
    freqs = [f1, f2]

    # create inputs and targets
    for trial in range(trials):

        # choose random delay
        delay = np.random.randint(low=delay_min, high=delay_max)
        trial_dur = evidence + delay + response

        # initialize arrays
        trial_inp = np.random.randn(trial_dur, 2) * noise
        trial_targ = np.zeros((trial_dur, 1))

        # choose random input channel
        idx = np.random.randint(low=0, high=2)
        f = freqs[idx]
        sine = np.sin(2.0 * np.pi * f * np.linspace(0.0, evidence * dt, evidence))
        trial_inp[:evidence, 0] += sine
        trial_inp[evidence+delay:, -1] += 1.0
        trial_targ[evidence+delay:, 0] = amp if idx == 0 else -amp

        # save trial inputs and targets
        inputs.append(trial_inp)
        targets.append(trial_targ)

    return inputs, targets


def frequency_matching(frequencies: np.ndarray, trials: int, evidence: int, delay_min: int, delay_max: int,
                       response: int, dt: float, noise: float) -> tuple:

    # allocate arrays
    inputs, targets = [], []
    n_freqs = len(frequencies)

    # create inputs and targets
    for trial in range(trials):

        # choose random delay
        delay = np.random.randint(low=delay_min, high=delay_max)
        trial_dur = evidence + delay + response

        # initialize arrays
        trial_inp = np.random.randn(trial_dur, 2) * noise
        trial_targ = np.zeros((trial_dur, 1))

        # choose random input channel
        idx = np.random.randint(low=0, high=n_freqs)
        f = frequencies[idx]
        sine = np.sin(2.0 * np.pi * f * np.linspace(0.0, trial_dur * dt, trial_dur))
        trial_inp[:evidence, 0] += sine[:evidence]
        trial_inp[evidence+delay:, -1] += 1.0
        trial_targ[evidence+delay:, 0] = sine[evidence+delay:]

        # save trial inputs and targets
        inputs.append(trial_inp)
        targets.append(trial_targ)

    return inputs, targets


def feature_choice(features: int, target_feature: int, trials: int, evidence: int, delay_min: int, delay_max: int,
                   response: int, noise: float, max_iter: int = 10) -> tuple:

    # define stimuli
    feature = [0, 1]
    stimuli = np.asarray(list(permutations(feature, features)))
    input_classes = len(stimuli)

    # create inputs and targets
    inputs, targets = [], []
    for trial in range(trials):

        # choose random delay
        delay = np.random.randint(low=delay_min, high=delay_max)
        trial_dur = evidence + delay + response

        # initialize arrays
        trial_inp = np.random.randn(trial_dur, features + 1) * noise
        trial_targ = np.zeros((trial_dur, 1))

        # choose random input class
        c = np.random.randint(low=0, high=input_classes)
        trial_inp[:evidence, :features] += stimuli[c]
        trial_inp[evidence+delay:, -1] += 1.0
        trial_targ[evidence+delay:, 0] = 1.0 if stimuli[c][target_feature] > 0.5 else -1.0

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


