import numpy as np
from scipy.signal import sosfiltfilt, butter, find_peaks
from scipy.optimize import minimize
import scipy.interpolate as interp
# TODO: Remove matplotlib later (but not EngFormatter)
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
# TODO: Delete time later
import time
from ..config import config
from ..plot import plot

def run(data,
        scale=None,
        delay_value=None,
        max_thz_frequency=50e12,
        recording_type="multi_cycle",
        filter_position=True,
        lowcut_position=None,
        highcut_position=100,
        filter_signal=True,
        lowcut_signal=1,
        highcut_signal=None,
        consider_all_traces=False,
        dataset_name=None,
        debug=False):
    if debug:
        config.set_debug(True)
    else:
        config.set_debug(False)
    data["scale"] = scale
    data["delay_value"] = delay_value
    data["number_of_traces"] = None
    data["interpolation_resolution"] = None
    data["trace_cut_index"] = None

    # Timestep in lab time
    data["dt"] = (data["time"][-1] - data["time"][0]) / (len(data["time"]) - 1)
    if debug and dataset_name == "light":
        fig, ax = plot.debug_lab_time_raw(data)
    if filter_position:
        data["position"] = butter_filter(data["position"],
                                         1 / data["dt"],
                                         lowcut=lowcut_position,
                                         highcut=highcut_position)
        if lowcut_position is not None and highcut_position is None:
            config.logger.info(f"Position data is high-pass filtered with {EngFormatter('Hz')(lowcut_position)}.")
        elif lowcut_position is None and highcut_position is not None:
            config.logger.info(f"Position data is low-pass filtered with {EngFormatter('Hz')(highcut_position)}.")
        elif lowcut_position is not None and highcut_position is not None:
            config.logger.info(
                f"Position data is low- and high-pass filtered with [{EngFormatter('Hz')(lowcut_position)}, {EngFormatter('Hz')(lowcut_position)}].")
    # Calculate the total record length in THz time, afterward we can select the correct interpolation resolution
    data["thz_recording_length"] = data["scale"] * (np.max(data["position"]) - np.min(data["position"]))
    data["thz_start_offset"] = data["scale"] * np.min(data["position"])
    if filter_signal:
        if highcut_signal is None:
            highcut_signal = max_thz_frequency
        data["signal"] = butter_filter(data["signal"],
                                       1 / data["dt"],
                                       lowcut=lowcut_signal,
                                       highcut=highcut_signal)

        if lowcut_signal is not None and highcut_signal is None:
            config.logger.info(f"Signal data is high-pass filtered with {EngFormatter('Hz')(lowcut_signal)}.")
        elif lowcut_signal is None and highcut_signal is not None:
            config.logger.info(f"Signal data is low-pass filtered with {EngFormatter('Hz')(highcut_signal)}.")
        elif lowcut_signal is not None and highcut_signal is not None:
            config.logger.info(
                f"Signal data is low- and high-pass filtered with [{EngFormatter('Hz')(lowcut_signal)}, {EngFormatter('Hz')(highcut_signal)}].")
    if filter_signal:
        data = resample_data(data, max_thz_frequency)
    if recording_type == "single_cycle":
        if np.argmin(data["position"]) - np.argmax(data["position"]) < 0:
            # Either first minimum, then maximum
            data["trace_cut_index"] = np.array([np.argmin(data["position"]), np.argmax(data["position"])])
        else:
            # Otherwise, first maximum, then minimum
            data["trace_cut_index"] = np.array([np.argmax(data["position"]), np.argmin(data["position"])])
    else:
        # Get the peaks of the sinusoid (or similar) of the position data, then we know the number of traces
        data = get_multiple_index(data, filter_position, highcut_position)
    if debug and dataset_name == "light":
        fig, ax = plot.debug_lab_time_filtered(data, lowcut_position, highcut_position, lowcut_signal, highcut_signal,
                                               fig, ax)
        fig2, ax2 = plot.debug_position_cut(data, dataset_name)
    data = cut_incomplete_traces(data)
    for exponent in range(6, 20):
        if 0.5 * (2 ** exponent / data["thz_recording_length"]) > max_thz_frequency:
            data["interpolation_resolution"] = 2 ** exponent
            config.logger.info(
                "Found interpolation resolution to have more than " +
                f"{EngFormatter('Hz', places=1)(max_thz_frequency)}: 2 ** {exponent} = " +
                f"{2 ** exponent} points")
            break
    if data["interpolation_resolution"] is None:
        raise ValueError("Could not find a proper interpolation resolution between 2**6 and 2**20."
                         "Did you select the right scale [ps/V] and the right max_THz_frequency?")
    # If we recorded multiple forward/backward traces, we can calculate the delay between position and signal.
    if recording_type == "multi_cycle":
        original_time = np.arange(0, data["dt"] * len(data["position"]), data["dt"])
        original_time = original_time[:data["position"].size]
        position_interpolated = interp.interp1d(original_time,
                                                data["position"],
                                                bounds_error=False,
                                                fill_value=np.nan)
        interpolated_delay = np.linspace(0, 1, data["interpolation_resolution"])
        data["light_time"] = interpolated_delay * data["thz_recording_length"] + data["thz_start_offset"]
        # Get timedelay between position and signal
        if debug and dataset_name == "light":
            fig3, ax3 = plot.debug_no_delay_compensation(data, original_time, position_interpolated, interpolated_delay,
                                                         consider_all_traces)
        if data["delay_value"] is None:
            config.logger.info(
                f"No delay_value provided, searching now for optimal delay:")
            data["delay_value"] = get_delay(data, original_time, position_interpolated, consider_all_traces, debug)
        shift_position(data, original_time, position_interpolated)
        if debug and dataset_name == "light":
            fig3, ax3 = plot.debug_with_delay_compensation(data, position_interpolated, interpolated_delay,
                                                           consider_all_traces, dataset_name, fig3, ax3)
    return data


def resample_data(data, max_thz_frequency):
    """This is a little bit tricky, since we have the sampling time in lab time but not in "light time" [ps].
    The self.max_THz_frequency is defined in the time frame of the THz sample.

    The max. slope of the position data vs. lab time is the smallest max. THz frequency
    """
    # TODO: Needs to be checked
    # TODO: Currently just skipping values by a factor, we can improve the signal to
    #  first low-pass filter it, then take the larger steps
    # [V/s] * [ps/V] --> scaling factor
    max_native_frequency = 1 / (
            np.max(np.gradient(data["position"], data["dt"])) * data["scale"] * data["dt"])
    factor = np.int64(np.floor(max_native_frequency / max_thz_frequency))
    if factor < 1:
        config.logger.debug(f"No resampling necessary.")
        return data
    current_time = np.arange(0, len(data["position"]) * data["dt"], data["dt"])
    new_dt = factor * data["dt"]
    new_time = np.arange(0, len(data["position"]) * data["dt"], new_dt)
    config.logger.info(
        f"Current time sample: {EngFormatter('s')(data['dt'])} per sample. New time sample: {EngFormatter('s')(new_dt)} per sample.")
    data["position"] = np.interp(new_time, current_time, data["position"])
    data["signal"] = np.interp(new_time, current_time, data["signal"])
    data["dt"] = new_dt
    return data


def get_multiple_index(data, filter_position, highcut_position):
    position = data["position"] - np.mean(data["position"])
    if filter_position and highcut_position is not None:
        original_max_freq = 1 / data["dt"]
        new_max_freq = 10 * highcut_position
        reduction_factor = int(np.round(original_max_freq / new_max_freq))
        signal_fft = np.abs(np.fft.rfft(np.abs(position[::reduction_factor])))
        freq = np.fft.rfftfreq(len(position[::reduction_factor]), reduction_factor * data["dt"])
        # Excluding the zero frequency "peak", which is related to offset
        guess_freq = np.abs(freq[np.argmax(signal_fft[1:]) + 1])
    else:
        start = time.time()
        signal_fft = np.abs(np.fft.rfft(np.abs(position)))
        freq = np.fft.rfftfreq(len(position), data["dt"])
        guess_freq = np.abs(freq[np.argmax(signal_fft[1:]) + 1])
        end = time.time()
        config.logger.info(
            f"Taking rFFT over complete position array, taking {EngFormatter('s', places=1)(end - start)}." +
            "If you specify filter_position=True and a reasonable highcut_position (in [Hz]), " +
            "you can accelerate this process alot.")

    idx, _ = find_peaks(np.abs(position),
                        height=0.8 * np.max(np.abs(position)),
                        distance=round(0.9 * (1 / guess_freq) / data["dt"]))
    data["trace_cut_index"] = idx
    data["number_of_traces"] = len(data["trace_cut_index"]) + 1
    return data


def cut_incomplete_traces(data):
    """Cut any incomplete trace from the array before the first delay peak or after the last delay peak"""
    data["position"] = data["position"][data["trace_cut_index"][0]:data["trace_cut_index"][-1]]
    data["signal"] = data["signal"][data["trace_cut_index"][0]:data["trace_cut_index"][-1]]
    data["trace_cut_index"] = data["trace_cut_index"][1:-1] - data["trace_cut_index"][0]
    data["number_of_traces"] = len(data["trace_cut_index"]) + 1
    return data


def get_delay(data, original_time, position_interpolated, consider_all_traces, debug=False):
    interpolated_delay = np.linspace(0, 1, data["interpolation_resolution"])
    x0 = [0]
    init_simplex = np.array([0, 50]).reshape(2, 1)
    xatol = 0.1
    res = minimize(_minimize,
                   x0,
                   method="Nelder-Mead",
                   args=(data, original_time, position_interpolated, interpolated_delay, consider_all_traces),
                   options={"disp": debug,
                            "maxiter": 30,
                            "xatol": xatol,
                            "initial_simplex": init_simplex})
    return res.x[0]


def _minimize(delay, data, original_time, position_interpolated, interpolated_delay, consider_all_traces):
    new_time_axis = delay * data["dt"] + np.copy(original_time)
    new_position = position_interpolated(new_time_axis)
    new_position = (new_position - np.nanmin(new_position)) / (np.nanmax(new_position) - np.nanmin(new_position))
    signal_norm = (data["signal"] - np.min(data["signal"])) / (
            np.max(data["signal"]) - np.min(data["signal"]))
    signal_matrix = np.zeros((data["interpolation_resolution"], data["number_of_traces"]))
    signal_matrix[:] = np.NaN

    traces_for_testing = zip(np.split(new_position, data["trace_cut_index"]),
                             np.split(signal_norm, data["trace_cut_index"]))
    i = 0
    for position, signal in traces_for_testing:
        # Numpy's interpolation method needs sorted, strictly increasing values
        signal = signal[np.argsort(position)]
        position = position[np.argsort(position)]
        # Since it needs to be strictly increasing, keep only values where x is strictly increasing.
        # Ignore any other y value when it has the same x value.
        signal = np.append(signal[0], signal[1:][(np.diff(position) > 0)])
        position = np.append(position[0], position[1:][(np.diff(position) > 0)])

        signal = np.interp(interpolated_delay, position, signal)
        signal_matrix[:, i] = signal
        if not consider_all_traces and i > 100:
            break
        i += 1
    config.logger.info(f"Delay:\t{delay[0]:.3f}\tError:\t{np.sum(np.nanstd(signal_matrix, axis=1))}")
    return np.sum(np.nanstd(signal_matrix, axis=1))


def shift_position(data, original_time, position_interpolated):
    config.logger.info(
        f"Found optimal delay_value {EngFormatter('Sa', places=3)(data['delay_value'])}, corresponding to a time delay of {EngFormatter('s')(data['delay_value'] * data['dt'])}.")
    new_time_axis = data["delay_value"] * data["dt"] + np.copy(original_time)
    data["position"] = position_interpolated(new_time_axis)
    data["signal"] = data["signal"][~np.isnan(data["position"])]
    data["position"] = data["position"][~np.isnan(data["position"])]


def butter_filter(data, fs, lowcut=None, highcut=None, order=5):
    sos = _butter_coeff(fs, lowcut, highcut, order=order)
    y = sosfiltfilt(sos, data, padtype=None)
    return y


def _butter_coeff(fs, lowcut=None, highcut=None, order=None):
    """Create coefficients for a butterworth filter."""
    nyq = 0.5 * fs
    if highcut > nyq:
        config.logger.info(
            f"{EngFormatter('Hz')(highcut)} > Nyquist-frequency ({EngFormatter('Hz')(nyq)}), "
            "ignoring parameter.")
        highcut = None
    if lowcut is not None and highcut is not None:
        # Bandpass filter
        if lowcut > highcut:
            raise ValueError(
                f"Lowcut is bigger than highcut! {EngFormatter('Hz')(lowcut)} > {EngFormatter('Hz')(nyq)}")
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    elif highcut is not None:
        # Low pass filter
        low = highcut / nyq
        sos = butter(order, low, analog=False, btype='low', output='sos')
    elif lowcut is not None:
        # High pass filter
        high = lowcut / nyq
        sos = butter(order, high, analog=False, btype='high', output='sos')
    else:
        raise NotImplementedError("Lowcut and highcut need to be specified either with a frequency or 'None'.")
    return sos
