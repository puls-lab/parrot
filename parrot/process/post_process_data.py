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
        for mode in self.data.keys():
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
        for mode in self.data.keys():
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
        for mode in self.data.keys():
            time = self.data[mode]["light_time"]
            dt = (time[-1] - time[0]) / (len(time) - 1)
            self.data[mode]["frequency"] = np.fft.rfftfreq(len(time), dt)
            self.data[mode]["average"]["frequency_domain"] = np.fft.rfft(self.data[mode]["average"]["time_domain"])
        return self.data

    def pad_zeros(self, new_frequency_resolution=5e9):
        if "window" in self.applied_functions:
            if "light" in self.data.keys():
                raise NotImplementedError("Light data missing. You can only pad zeros to light data.")
            else:
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
                new_time = np.arange(current_time[0],
                                     current_time[0] + self.data["light"]["interpolation_resolution"] * dt,
                                     dt)
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
        elif "dark" not in self.data.keys():
            raise NotImplementedError("Dark trace missing. To create the polynomial, a dark trace is missing.")
        else:
            time = self.data["dark"]["light_time"]
            z = np.polyfit(time, self.data["dark"]["average"]["time_domain"], order)
            p = np.poly1d(z)
            for mode in self.data.keys():
                self.data[mode]["single_traces"] -= p(self.data[mode]["light_time"]).reshape(-1, 1)
                self.data[mode]["average"]["time_domain"] -= p(self.data[mode]["light_time"])
            self.applied_functions.append("subtract_polynomial")
            return self.data

    def correct_systematic_errors(self):
        # This only works when two dark traces were recorded with the same settings as with the light trace
        if "window" in self.applied_functions:
            raise NotImplementedError("You already applied a window to the data, "
                                      "you first have to subtract a polynomial and then apply a window.")
        elif "FFT" in self.applied_functions:
            raise NotImplementedError("You already applied a FFT to the data, "
                                      "you first have to subtract a polynomial and do a FFT.")
        elif "pad_zeros" in self.applied_functions:
            raise NotImplementedError("You already applied zero-padding to the data, "
                                      "you first have to subtract a polynomial and then pad_zeros.")
        elif "dark1" not in self.data.keys() and "dark2" not in self.data.keys():
            raise NotImplementedError("Two dark traces missing.")
        else:
            dark1_avg = self.data["dark1"]["average"]["time_domain"]
            dark2_avg = self.data["dark2"]["average"]["time_domain"]
            min_number_traces = np.min(np.array([self.data["dark1"]["single_traces"].shape[1],
                                                 self.data["dark2"]["single_traces"].shape[1]]))
            self.data["dark"] = {"average": {}}
            self.data["dark"]["light_time"] = self.data["dark1"]["light_time"]
            self.data["dark"]["number_of_traces"] = min_number_traces
            self.data["dark"]["single_traces"] = (self.data["dark1"]["single_traces"][:, :min_number_traces]
                                                  - self.data["dark2"]["single_traces"][:, :min_number_traces])
            self.data["dark"]["average"]["time_domain"] = dark1_avg - dark2_avg
            self.data["light"]["single_traces"] -= dark1_avg.reshape(-1, 1)
            self.data["light"]["average"]["time_domain"] -= dark1_avg
            self.applied_functions.append("correct_systematic_errors")
            return self.data

    def get_statistics(self):
        """
        Basic definition of Dynamic range and Signal-To-Noise ratio as defined in:

        > Mira Naftaly and Richard Dudley
        > Methodologies for determining the dynamic ranges and signal-to-noise ratios of terahertz time-domain spectrometers
        > Optics Letters Vol. 34, Issue 8, pp. 1213-1215 (2009)
        > https://doi.org/10.1364/OL.34.001213

        More detailed definition according to the VDI/VDE 5590 standard:
        > Time domain
        SNR(t) = mean(light_at_maximum(t)) / STD(light_at_maximum(t))
        DR(t) = ( mean(light_at_maximum(t)) - mean(noise_at_maximum_of_light(t)) ) / STD(noise_at_maximum_of_light(t))
        > Frequency domain
        SNR(f) = mean(light(f)) / STD(light(f))
        DR(f) = ( mean(light(f)) - mean(noise(f)) ) / STD(noise(f))
        """
        if "light" not in self.data.keys():
            raise NotImplementedError("No light-data detected, cannot calculate any meaningful SNR/DR.")
        else:
            self.data["statistics"] = {}
            # Signal-to-Noise ratio (SNR)
            # Time Domain
            # Calculate mean signal
            mean_light = np.mean(self.data["light"]["single_traces"], axis=1)
            # Extract index of peak location
            index = np.argmax(mean_light)
            std_of_peak = np.std(self.data["light"]["single_traces"][index, :])
            peak_snr_td = np.max(mean_light) / std_of_peak
            # Frequency Domain
            all_traces_fft = np.abs(np.fft.rfft(self.data["light"]["single_traces"], axis=0))
            std_light_fft = np.std(all_traces_fft, axis=1)
            mean_light_fft = np.abs(np.fft.rfft(np.mean(self.data["light"]["single_traces"], axis=1)))
            peak_snr_fd = np.max(mean_light_fft / std_light_fft)
            self.data["statistics"]["peak_SNR_time"] = peak_snr_td
            self.data["statistics"]["peak_SNR_freq"] = peak_snr_fd
        if "dark" in self.data.keys():
            # Dynamic Range (DR)
            # Time Domain
            mean_light = np.mean(self.data["light"]["single_traces"], axis=1)
            index = np.argmax(mean_light)
            mean_noise = np.mean(self.data["dark"]["single_traces"][index, :])
            std_noise = np.std(self.data["dark"]["single_traces"][index, :])
            peak_dr_td = (np.max(mean_light) - mean_noise) / std_noise
            # Frequency Domain
            mean_light_fft = np.abs(np.fft.rfft(np.mean(self.data["light"]["single_traces"], axis=1)))
            mean_dark_fft = np.abs(np.fft.rfft(np.mean(self.data["dark"]["single_traces"], axis=1)))
            std_dark_fft = np.std(np.abs(np.fft.rfft(self.data["dark"]["single_traces"], axis=0)), axis=1)
            peak_dr_fd = np.max((mean_light_fft - mean_dark_fft) / std_dark_fft)
            self.data["statistics"]["peak_DR_time"] = peak_dr_td
            self.data["statistics"]["peak_DR_freq"] = peak_dr_fd
        return self.data
