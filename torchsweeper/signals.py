import numpy as np
import torch as torch
from abc import ABC, abstractmethod


class Signal(ABC):

    @abstractmethod
    def __call__(self, n_samples) -> np.ndarray:
        raise NotImplementedError()


class NCyclePulse(Signal):

    def __init__(self, f, n_cycles, snr):
        self.d = n_cycles * 1 / f
        self.f = f
        self.snr = snr

    def __call__(self, n_samples):
        t = np.linspace(0, self.d, n_samples)
        w1 = np.hanning(len(t))
        w2 = np.linspace(1.0, 0.0, len(t)) ** 6.0
        clean_x = np.sin(2 * np.pi * self.f * t) * w1 * w2
        x = add_noise(clean_x, self.snr)
        return x / clean_x.max(), t


def add_noise(samples, snr):

    return samples + np.random.normal(
        scale=np.abs(samples).max() / snr, size=samples.size
        )


class SignalImage:

    def __init__(self, signal: Signal, stacking_shape: tuple, use: str='numpy', channels: int=1, batches: int=1):

        if use not in ['numpy', 'layer', 'transform']:
            raise ValueError("Use must be one of 'numpy', 'layer' or 'transform'")

        if use not in ['layer', 'transform'] and ((channels > 1) or (batches > 1)):
            raise ValueError("If SignalImage is to be used with numpy, channels and batches must both be set to 1.")

        if use == 'transform' and batches > 1:
            raise ValueError("If SignalImage is to be used with a monai transform, batches must be set to 1.")

        self.channels = channels
        self.batches = batches
        self.stacking_shape = stacking_shape
        self.use = use
        self.signal = signal

    def __call__(self, signal_len):

        def stack(stacking_shape):
            if len(stacking_shape) == 1:
                image = np.stack(
                    [self.signal(signal_len)[0] for _ in range(stacking_shape[0])]
                )
            else:
                image = stack(stacking_shape[:-1])
            return image

        np_image = stack(self.stacking_shape)
        if self.use == 'numpy':
            return np_image

        np_image_with_channels = np.stack([np_image] * self.channels)
        if self.use == 'transform':
            return np_image_with_channels

        if self.use == 'layer':
            return torch.as_tensor(np.ascontiguousarray(
                np.stack([np_image_with_channels] * self.batches)
            ), device=torch.device('cuda'))


class LengthSweepImageSet:

    def __init__(self, image: SignalImage, min_len, max_len, len_base=2):
        self.signal_lengths = [len_base ** (x + min_len) for x in range(max_len - min_len)]
        self.image = image

    def __call__(self):
        return [self.image(signal_length) for signal_length in self.signal_lengths]


