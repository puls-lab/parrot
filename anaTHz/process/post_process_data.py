import numpy as np
from scipy.signal.windows import flattop
from matplotlib.ticker import EngFormatter
import copy

class PostProcessData:
    def __init__(self, data, debug=True):
        self.data = copy.deepcopy(data)
        self.applied_functions = []
        self.debug = debug
        # TODO: FFT, Cutting, Windowing, Zero-padding

    def super_gaussian(self, window_width=0.8, window_shift=0, window_order=10):
        for mode in ["dark", "light"]:
            signal = self.data[mode]["average"]["time_domain"]
            win_shift = window_shift * len(signal)
            win_width = window_width * len(signal)
            tau = np.arange(0, len(signal))
            window = np.exp(
                -2 ** window_order * np.log(2) * np.abs(
                    (tau - (len(signal) - 1) / 2 - win_shift) / win_width) ** window_order)
            self.data[mode]["single_traces"] *= window.reshape(-1, 1)
            self.data[mode]["average"]["time_domain"] *= window
        self.applied_functions.append("window")
        return self.data

    def flat_top(self):
        for mode in ["dark", "light"]:
            signal = self.data[mode]["average"]["time_domain"]
            window = flattop(len(signal))
            self.data[mode]["single_traces"] *= window.reshape(-1, 1)
            self.data[mode]["average"]["time_domain"] *= window
        self.applied_functions.append("window")
        return self.data

    def calc_fft(self):
        if "window" not in self.applied_functions:
            print("INFO: You are taking the FFT without windowing the data, which could create artifacts "
                  "in frequency domain. It is strongly recommended to first window the data.")
        for mode in ["dark", "light"]:
            time = self.data[mode]["light_time"]
            dt = (time[-1] - time[0]) / (len(time) - 1)
            self.data[mode]["frequency"] = np.fft.rfftfreq(len(time), dt)
            self.data[mode]["average"]["frequency_domain"] = np.fft.rfft(self.data[mode]["average"]["time_domain"])
        return self.data

    def pad_zeros(self, new_frequency_resolution=5e9):
        if "window" in self.applied_functions:
            # TODO: Make pad_zeros in terms of 2 ** x for easier FFT
            current_time = self.data["light"]["light_time"]
            signal = self.data["light"]["average"]["time_domain"]
            dt = (current_time[-1] - current_time[0]) / (len(current_time) - 1)
            current_td_length = np.abs(current_time[-1] - current_time[0])
            new_td_length = 1 / new_frequency_resolution
            max_THz_frequency = len(current_time) / current_td_length
            new_interpolation_resolution = None
            for exponent in range(6, 21):
                if 0.5 * (2 ** exponent / new_td_length) > max_THz_frequency:
                    new_interpolation_resolution = 2 ** exponent
                    if self.debug:
                        print("INFO: Found interpolation resolution to have more than "
                              f"{EngFormatter('Hz', places=1)(max_THz_frequency)}: 2 ** {exponent} = "
                              f"{2 ** exponent} points")
                    break
            if new_interpolation_resolution is None:
                raise ValueError("Could not find a proper interpolation resolution between 2**6 and 2**21."
                                 "Is your new frequency resolution too low?")
            self.data["light"]["interpolation_resolution"] = new_interpolation_resolution
            padded_array = np.zeros(self.data["light"]["interpolation_resolution"])
            padded_array[:len(current_time)] = signal
            new_time = np.arange(current_time[0], current_time[0] + self.data["light"]["interpolation_resolution"] * dt, dt)
            self.data["light"]["light_time"] = new_time
            self.data["light"]["average"]["time_domain"] = padded_array
            matrix = np.zeros((len(new_time), self.data["light"]["single_traces"].shape[1]))
            matrix[:len(current_time), :] = self.data["light"]["single_traces"]
            self.data["light"]["single_traces"] = matrix
            self.applied_functions.append("pad_zeros")
        else:
            raise NotImplementedError("You need to first window the data before padding zeros.")
        return self.data

    def subtract_polynomial(self, order=2):
        if "window" in self.applied_functions:
            raise NotImplementedError("You already applied a window to the data, "
                                      "you first have to subtract a polynomial and then apply a window.")
        elif "FFT" in self.applied_functions:
            raise NotImplementedError("You already applied a FFT to the data, "
                                      "you first have to subtract a polynomial and do a FFT.")
        elif "pad_zeros" in self.applied_functions:
            raise NotImplementedError("You already applied zero-padding to the data, "
                                      "you first have to subtract a polynomial and then pad_zeros.")
        else:
            time = self.data["dark"]["light_time"]
            z = np.polyfit(time, self.data["dark"]["average"]["time_domain"], order)
            p = np.poly1d(z)
            for mode in ["dark", "light"]:
                self.data[mode]["single_traces"] -= p(self.data[mode]["light_time"]).reshape(-1, 1)
                self.data[mode]["average"]["time_domain"] -= p(self.data[mode]["light_time"])
            self.applied_functions.append("subtract_polynomial")
            return self.data
