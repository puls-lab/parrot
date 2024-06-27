import numpy as np
from scipy.signal.windows import flattop
from matplotlib.ticker import EngFormatter
import copy

import logging

logger = logging.getLogger(__name__)
logger.handlers.clear()

formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    Authors: Joe Kington & David Parks
    Source:  https://stackoverflow.com/a/4495197/8599759
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx

def _calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft

class PostProcessData:
    def __init__(self, data, debug=True):
        self.data = copy.deepcopy(data)
        self.applied_functions = []
        self.debug = debug
        # TODO: FFT, Cutting, Windowing, Zero-padding

        self.debug = debug
        if not debug:
            logger.setLevel(logging.WARNING)

    def super_gaussian(self, window_width=0.8, window_shift=0, window_order=10):
        for mode in self.data.keys():
            if mode == "light" or mode == "dark1" or mode == "dark2" or mode == "dark":
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
            logger.warning(
                  "You are taking the FFT without windowing the data, which could create artifacts "
                  "in frequency domain. It is strongly recommended to first window the data.")
        for mode in self.data.keys():
            time = self.data[mode]["light_time"]
            dt = (time[-1] - time[0]) / (len(time) - 1)
            self.data[mode]["frequency"] = np.fft.rfftfreq(len(time), dt)
            self.data[mode]["average"]["frequency_domain"] = np.fft.rfft(self.data[mode]["average"]["time_domain"])
        return self.data

    def pad_zeros(self, new_frequency_resolution=5e9, min_test_exponent=6, max_test_exponent=21):
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
                for exponent in range(min_test_exponent, max_test_exponent):
                    if 0.5 * (2 ** exponent / new_td_length) > max_THz_frequency:
                        new_interpolation_resolution = 2 ** exponent
                        logger.info("Found interpolation resolution to have more than "
                                  f"{EngFormatter('Hz', places=1)(max_THz_frequency)}: 2 ** {exponent} = "
                                  f"{2 ** exponent} points")
                        break
                if new_interpolation_resolution is None:
                    raise ValueError(f"Could not find a proper interpolation resolution between 2**{min_test_exponent} "+
                                     f"and 2**{max_test_exponent}."
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

    def cut_data(self, time_start=None, time_stop=None):
        if time_start is None and time_stop is None:
            raise NotImplementedError("You need to supply either a start time, a stop time or both.")
        time = self.data["dark"]["light_time"]
        if time_start is None:
            time_start = np.min(time)
        if time_stop is None:
            time_stop = np.max(time)
        start_idx = np.where(time >= time_start)[0][0]
        stop_idx = np.where(time >= time_stop)[0][0]
        for mode in self.data.keys():
            if mode == "light" or mode == "dark1" or mode == "dark2" or mode == "dark":
                self.data[mode]["single_traces"] = self.data[mode]["single_traces"][start_idx:stop_idx, :]
                self.data[mode]["average"]["time_domain"] = self.data[mode]["average"]["time_domain"][start_idx:stop_idx]
                self.data[mode]["light_time"] = self.data[mode]["light_time"][start_idx:stop_idx]
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
                if mode == "light" or mode == "dark1" or mode == "dark2" or mode == "dark":
                    self.data[mode]["single_traces"] -= p(self.data[mode]["light_time"]).reshape(-1, 1)
                    self.data[mode]["average"]["time_domain"] -= p(self.data[mode]["light_time"])
            self.applied_functions.append("subtract_polynomial")
            return self.data

    def correct_systematic_errors(self):
        # This only works when two dark traces were recorded with the same settings as with the light trace
        if "window" in self.applied_functions:
            raise NotImplementedError("You already applied a window to the data, "
                                      "you first have to correct for systematic errors and then apply a window.")
        elif "FFT" in self.applied_functions:
            raise NotImplementedError("You already applied a FFT to the data, "
                                      "you first have to correct for systematic errors and then do a FFT.")
        elif "pad_zeros" in self.applied_functions:
            raise NotImplementedError("You already applied zero-padding to the data, "
                                      "you first have to correct for systematic errors and then pad zeros.")
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

    def correct_gain_in_spectrum(self):
        logger.warning(
            "This is an experimental feature!"
            "It should only affect the magnitude but not the phase. If you do spectroscopic experiments, please test by yourself."
            "The noise floor of the THz traces are flattened by creating a gain function based on RMS averaging of the single dark traces.")
        if "dark" not in self.data.keys():
            if "dark1" not in self.data.keys() and "dark2" not in self.data.keys():
                raise NotImplementedError("You submitted two dark traces. "
                                          "First, execute correct_systematic_errors() to generate a single, "
                                          "compensated dark trace. Afterwards, execute correct_gain_in_spectrum()")
            else:
                raise NotImplementedError(
                    "Dark traces is missing. To correct the gain in frequency domain, a dark trace is missing.")
        # Take FFT of each single dark trace without averaging them first
        dark_fft_single_traces = np.fft.rfft(self.data["dark"]["single_traces"], axis=0).T
        # RMS averaging of each single FFT dark trace
        smooth_gain = 1 / np.mean(np.abs(dark_fft_single_traces.T), axis=1)
        # Scale smooth_gain as such, that the noise floor is shifted to 1.
        factor = 1 / np.mean(np.abs(self.data["dark"]["average"]["frequency_domain"]) * smooth_gain)
        # Numpy throws an error when using *= notation, since we multiply float64 with complex128 dtypes.
        smooth_gain = smooth_gain * factor
        for mode in self.data.keys():
            self.data[mode]["average"]["frequency_domain"] = self.data[mode]["average"][
                                                                 "frequency_domain"] * smooth_gain
            # We have to permeate the changes also to the time domain, so that the data between freq domain and time
            # domain are consistent with each other
            self.data[mode]["average"]["time_domain"] = np.fft.irfft(self.data[mode]["average"]["frequency_domain"],
                                                                     n=len(self.data[mode]["average"]["time_domain"]))
            # Not only the averaged traces, but also the single traces are affected by the flattening of the noise
            # floor.
            single_traces = np.fft.rfft(self.data[mode]["single_traces"], axis=0).T
            single_traces = single_traces * smooth_gain
            self.data[mode]["single_traces"] = np.fft.irfft(single_traces.T, axis=0,
                                                            n=len(self.data[mode]["average"]["time_domain"])).T
        self.applied_functions.append("correct_gain_in_spectrum")
        return self.data

    def get_statistics(self, data):
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
        if "light" not in data.keys():
            raise NotImplementedError("No light-data detected, cannot calculate any meaningful SNR/DR.")
        else:
            data["statistics"] = {}
            # Signal-to-Noise ratio (SNR)
            # Time Domain
            # Calculate mean signal
            mean_light = np.mean(data["light"]["single_traces"], axis=1)
            # Extract index of peak location
            index = np.argmax(mean_light)
            std_of_peak = np.std(data["light"]["single_traces"][index, :])
            peak_snr_td = np.max(mean_light) / std_of_peak
            # Frequency Domain
            dt = (data["light"]["light_time"][-1] - data["light"]["light_time"][0]) / (
                        len(data["light"]["light_time"]) - 1)
            frequency = np.fft.rfftfreq(len(data["light"]["light_time"]), dt)
            all_traces_fft = np.abs(np.fft.rfft(data["light"]["single_traces"], axis=0))
            std_light_fft = np.std(all_traces_fft, axis=1)
            mean_light_fft = np.abs(np.fft.rfft(np.mean(data["light"]["single_traces"], axis=1)))
            peak_snr_fd = np.max(mean_light_fft / std_light_fft)
            # Calculate FWHM in frequency domain
            condition = (mean_light_fft / np.max(mean_light_fft)) > 0.5
            range_idx = 0
            for start, stop in contiguous_regions(condition):
                logger.info("Found segment above 0.5 (rel. amplitude), from \n" +
                            f"{EngFormatter('Hz')(frequency[start])} to {EngFormatter('Hz')(frequency[stop])} = {EngFormatter('Hz')(frequency[stop - 1] - frequency[start])}")
                if stop - start > range_idx:
                    range_idx = stop - start
                    data["statistics"]["fwhm_start"] = frequency[start]
                    data["statistics"]["fwhm_stop"] = frequency[stop - 1]
                    data["statistics"]["fwhm"] = frequency[stop - 1] - frequency[start]
            if range_idx == 0:
                logger.warning("Could not find Full-Width at Half-Maximum (FWHM).")
            data["statistics"]["peak_SNR_time"] = peak_snr_td
            data["statistics"]["peak_SNR_freq"] = peak_snr_fd
        if "dark" in data.keys():
            # Dynamic Range (DR)
            # Time Domain
            mean_light = np.mean(data["light"]["single_traces"], axis=1)
            index = np.argmax(mean_light)
            mean_noise = np.mean(data["dark"]["single_traces"][index, :])
            std_noise = np.std(data["dark"]["single_traces"][index, :])
            peak_dr_td = (np.max(mean_light) - mean_noise) / std_noise
            # Frequency Domain
            mean_light_fft = np.abs(np.fft.rfft(np.mean(data["light"]["single_traces"], axis=1)))
            mean_dark_fft = np.abs(np.fft.rfft(np.mean(data["dark"]["single_traces"], axis=1)))
            std_dark_fft = np.std(np.abs(np.fft.rfft(data["dark"]["single_traces"], axis=0)), axis=1)
            peak_dr_fd = np.max((mean_light_fft - mean_dark_fft) / std_dark_fft)
            data["statistics"]["peak_DR_time"] = peak_dr_td
            data["statistics"]["peak_DR_freq"] = peak_dr_fd
        return data

    def extract_bandwidth(self, data, min_THz_frequency=0e12, max_THz_frequency=10e12, threshold_dB=10):
        """Extract the bandwidth of the power spectrum, displayed on a logarithmic plot.
        """
        # Find frequency range, where averaged light traces in frequency domain are
        # at least threshold_dB above the averaged dark trace.
        frequency_dark, signal_fft_dark = _calc_fft(data["dark"]["light_time"],
                                                   data["dark"]["average"]["time_domain"])
        frequency_light, signal_fft_light = _calc_fft(data["light"]["light_time"],
                                                     data["light"]["average"]["time_domain"])
        filter_frequency_dark = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)
        filter_frequency_light = (frequency_light >= min_THz_frequency) & (
                frequency_light <= max_THz_frequency)
        if np.all(np.diff(frequency_dark[filter_frequency_dark]) > 0):
            power_dark = np.interp(frequency_light[filter_frequency_light], frequency_dark[filter_frequency_dark],
                                   np.abs(signal_fft_dark[filter_frequency_dark]) ** 2)
            power_light = np.abs(signal_fft_light[filter_frequency_light]) ** 2
            frequency = frequency_light[filter_frequency_light]
        else:
            raise ValueError("The frequency axis is not strictly increasing, "
                             "which is a necessity for numpy's interpolation function.")
        condition = (10 * np.log10(power_light) - 10 * np.log10(power_dark)) > threshold_dB
        range_idx = 0
        for start, stop in contiguous_regions(condition):
            logger.info(f"Found segment above {threshold_dB} dB, from \n" +
                        f"{EngFormatter('Hz')(frequency[start])} to {EngFormatter('Hz')(frequency[stop - 1])} = {EngFormatter('Hz')(frequency[stop - 1] - frequency[start])}")
            if stop - start > range_idx:
                range_idx = stop - start
                data["statistics"]["bandwidth_start"] = frequency[start]
                data["statistics"]["bandwidth_stop"] = frequency[stop - 1]
                data["statistics"]["bandwidth"] = frequency[stop - 1] - frequency[start]
                data["statistics"]["bandwidth_threshold_dB"] = threshold_dB
        if range_idx == 0:
            logger.warning(f"Could not find bandwidth above {threshold_dB} dB.")
        return data

