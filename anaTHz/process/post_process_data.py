import numpy as np


class PostProcessData:
    def __init__(self, signal, win_width=0.8, win_shift=0, win_order=10):
        self.signal = signal
        self.win_width = win_width
        self.win_shift = win_shift
        self.win_order = win_order
        # TODO: FFT, Cutting, Windowing, Zero-padding

    def super_gaussian(self):
        win_shift = self.win_shift * len(self.signal)
        win_width = self.win_width * len(self.signal)
        tau = np.arange(0, len(self.signal))
        window = np.exp(
            -2 ** self.win_order * np.log(2) * np.abs(
                (tau - (len(self.signal) - 1) / 2 - win_shift) / win_width) ** self.win_order)
        return window
