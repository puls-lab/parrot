"""Module for preparing the entered THz data

The sampling rate is checked and various low and/or high-pass filter are applied.
The number of traces is extracted and the positional data is split according to the extrema in the position signal.
As a highlight of this module, possible phase delays between position signal and THz signal are compensated for.
The data is first given back to the "proces_data.py" module before being returned to the user.
"""
import numpy as np
import scipy.signal
from scipy.signal import sosfiltfilt, butter, find_peaks, decimate
from scipy.optimize import minimize, direct
import scipy.interpolate as interp
# Only for debugging, otherwise PyCharm crashes
# import matplotlib
# matplotlib.use('TKAgg')
# import matplotlib.pyplot as plt
from timeit import default_timer as timer

from matplotlib.ticker import EngFormatter
from ..config import config
from ..plot import plot_debug


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
        resample=True,
        consider_all_traces=False,
        global_delay_search=True,
        search_radius=None,
        local_search_startpoint=50,
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
        fig, ax = plot_debug.lab_time_raw(data)
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
                f'Position data is low- and high-pass filtered with [{EngFormatter("Hz")(lowcut_position)}, '
                f'{EngFormatter("Hz")(lowcut_position)}].')
    # Calculate the total record length in THz time, afterward we can select the correct interpolation resolution
    data["thz_recording_length"] = data["scale"] * (np.max(data["position"]) - np.min(data["position"]))
    data["thz_start_offset"] = data["scale"] * np.min(data["position"])
    if filter_signal:
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
                f'Signal data is low- and high-pass filtered with [{EngFormatter("Hz")(lowcut_signal)}, '
                f'{EngFormatter("Hz")(highcut_signal)}].')
    if filter_signal and resample:
        data = resample_data(data, max_thz_frequency)
    if recording_type == "single_cycle":
        if np.argmin(data["position"]) - np.argmax(data["position"]) < 0:
            # Either first minimum, then maximum
            data["trace_cut_index"] = np.array([np.argmin(data["position"]), np.argmax(data["position"])])
        else:
            # Otherwise, first maximum, then minimum
            data["trace_cut_index"] = np.array([np.argmax(data["position"]), np.argmin(data["position"])])
            data["number_of_traces"] = 1
    else:
        # Get the peaks of the sinusoid (or similar) of the position data, then we know the number of traces
        trace_cut_index, number_of_traces = get_multiple_index(data["dt"], data["position"], filter_position,
                                                               highcut_position)
        data["trace_cut_index"] = trace_cut_index
        data["number_of_traces"] = number_of_traces
    if debug and dataset_name == "light":
        fig, ax = plot_debug.lab_time_filtered(data, lowcut_position, highcut_position, lowcut_signal, highcut_signal,
                                               fig, ax)
        fig2, ax2 = plot_debug.position_cut(data, dataset_name)
    data = cut_incomplete_traces(data)
    for exponent in range(6, 20):
        if 0.5 * (2 ** exponent / data["thz_recording_length"]) > max_thz_frequency:
            data["interpolation_resolution"] = 2 ** exponent
            config.logger.info(
                "Found interpolation resolution to have more than " +
                f"{EngFormatter('Hz', places=1)(max_thz_frequency)}: 2 ** {exponent} = " +
                f"{2 ** exponent} points.")
            break
    if data["interpolation_resolution"] is None:
        raise ValueError("Could not find a proper interpolation resolution between 2**6 and 2**20."
                         "Did you select the right scale [ps/V] and the right max_THz_frequency?")
    # If we recorded multiple forward/backward traces, we can calculate the delay between position and signal.
    if recording_type == "multi_cycle":
        original_time = np.arange(0, data["dt"] * len(data["position"]), data["dt"])
        original_time = original_time[:data["position"].size]
        signal_interpolated = interp.interp1d(original_time,
                                              data["signal"],
                                              bounds_error=False,
                                              fill_value=np.nan)
        interpolated_delay = np.linspace(0, 1, data["interpolation_resolution"])
        data["light_time"] = interpolated_delay * data["thz_recording_length"] + data["thz_start_offset"]
        # Get timedelay between position and signal
        if debug and dataset_name == "light":
            old_signal = signal_interpolated(original_time)
            data["temp"] = {"split_pos": np.split(data["position"], data["trace_cut_index"]),
                            "split_sig": np.split(data["signal"], data["trace_cut_index"]),
                            "old_signal": old_signal}
        if data["delay_value"] is None:
            config.logger.info(
                f"No delay_value provided, searching now for optimal delay... (please wait)")
            if global_delay_search:
                data["delay_value"] = get_global_delay(data, original_time, signal_interpolated,
                                                       consider_all_traces, search_radius, debug)
            else:
                data["delay_value"] = get_local_delay(data, original_time, signal_interpolated,
                                                      consider_all_traces, local_search_startpoint, debug)
        shift_position(data, original_time, signal_interpolated)
        if debug and dataset_name == "light":
            fig3, ax3 = plot_debug.with_delay_compensation(data, interpolated_delay, consider_all_traces, dataset_name)
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
    if factor < 2:
        config.logger.debug(f"No resampling necessary.")
        return data
    current_time = np.arange(0, len(data["position"]) * data["dt"], data["dt"])
    current_sampling_rate = 1 / data["dt"]

    new_dt = factor * data["dt"]
    new_time = np.arange(0, len(data["position"]) * data["dt"], new_dt)
    new_sampling_rate = 1 / new_dt
    position_filtered = butter_filter(data["position"], current_sampling_rate, highcut=new_sampling_rate)
    signal_filtered = butter_filter(data["signal"], current_sampling_rate, highcut=new_sampling_rate)

    config.logger.info(
        f"Current time sample: {EngFormatter('s')(data['dt'])} per sample. New time sample: {EngFormatter('s')(new_dt)} per sample. Factor: {factor}x")
    # Large arrays can sometimes have round-off errors in length by +-1 (e.g. after butter_filter).
    # To rectify this, cut all arrays to their common, minimum length
    min_length = np.min(np.array([len(current_time), len(position_filtered), len(signal_filtered)]))
    current_time = current_time[:min_length]
    position_filtered = position_filtered[:min_length]
    signal_filtered = signal_filtered[:min_length]
    data["position"] = np.interp(new_time, current_time, position_filtered)
    data["signal"] = np.interp(new_time, current_time, signal_filtered)
    data["time"] = new_time
    data["dt"] = new_dt
    return data


def get_multiple_index(dt, position, filter_position, highcut_position):
    position = position - np.mean(position)
    if filter_position and highcut_position is not None:
        original_max_freq = 1 / dt
        new_max_freq = 10 * highcut_position
        reduction_factor = int(np.round(original_max_freq / new_max_freq))
        signal_fft = np.abs(np.fft.rfft(np.abs(position[::reduction_factor])))
        freq = np.fft.rfftfreq(len(position[::reduction_factor]), reduction_factor * dt)
        # Excluding the zero frequency "peak", which is related to offset
        guess_freq = np.abs(freq[np.argmax(signal_fft[1:]) + 1])
    else:
        start = timer()
        signal_fft = np.abs(np.fft.rfft(np.abs(position)))
        freq = np.fft.rfftfreq(len(position), dt)
        guess_freq = np.abs(freq[np.argmax(signal_fft[1:]) + 1])
        end = timer()
        config.logger.info(
            f"Taking rFFT over complete position array, taking {EngFormatter('s', places=1)(end - start)}." +
            "If you specify filter_position=True and a reasonable highcut_position (in [Hz]), " +
            "you can accelerate this process alot.")

    idx, _ = find_peaks(np.abs(position),
                        height=0.8 * np.max(np.abs(position)),
                        distance=round(0.9 * (1 / guess_freq) / dt))
    trace_cut_index = idx
    number_of_traces = len(trace_cut_index) + 1
    return trace_cut_index, number_of_traces


def cut_incomplete_traces(data):
    """Cut any incomplete trace from the array before the first delay peak or after the last delay peak"""
    data["time"] = data["time"][data["trace_cut_index"][0]:data["trace_cut_index"][-1]]
    data["position"] = data["position"][data["trace_cut_index"][0]:data["trace_cut_index"][-1]]
    data["signal"] = data["signal"][data["trace_cut_index"][0]:data["trace_cut_index"][-1]]
    data["trace_cut_index"] = data["trace_cut_index"][1:-1] - data["trace_cut_index"][0]
    data["number_of_traces"] = len(data["trace_cut_index"]) + 1
    return data


def _housekeeping_optimizer(delay, error):
    global iteration_steps
    global iteration_delays
    global iteration_errors
    if len(iteration_steps) == 0:
        iteration_steps = [1]
    else:
        iteration_steps.append(iteration_steps[-1] + 1)
    iteration_delays.append(delay)
    iteration_errors.append(error)


def get_global_delay(data, original_time, signal_interpolated, consider_all_traces, search_radius=None, debug=False):
    """Tried various global optimization alogirithms from the SciPy package.
    The DIRECT algorithm was typically fast (< 10s) and reliable in finding the global minimum.

    consider_all_traces is False as standard.
    That means, that only the first 100 traces are considered when calculating the standard deviation.
    This should be normally sufficient and speed up computation."""
    global iteration_steps
    global iteration_delays
    global iteration_errors
    interpolated_delay = np.linspace(0, 1, data["interpolation_resolution"])
    # Extract length (in timesamples) for one full delay scan. Don't take the first index,
    # since it start position can be arbitrary
    half_position_cycle = np.median(np.diff(data["trace_cut_index"])) / 2  # In time samples
    bounds = [(-half_position_cycle, +half_position_cycle)]
    if search_radius is not None:
        bounds = [(-search_radius, +search_radius)]

    iteration_steps = []
    iteration_delays = []
    iteration_errors = []
    original_position_short = None
    original_time_short = None
    trace_cut_idx = None
    number_of_traces_tested = 102
    if not consider_all_traces and data["number_of_traces"] > number_of_traces_tested:
        #  Cut position and signal data to the first 100 traces to accelerate shifting of the arrays
        #  and using less memory
        original_time_short = np.copy(original_time)[
                              data["trace_cut_index"][1]:data["trace_cut_index"][number_of_traces_tested - 1]]
        trace_cut_idx = data["trace_cut_index"][(data["trace_cut_index"] >= data["trace_cut_index"][2]) &
                                                (data["trace_cut_index"] <= data["trace_cut_index"][
                                                    number_of_traces_tested - 2])]
        trace_cut_idx -= data["trace_cut_index"][1]
        original_position_short = np.copy(data["position"])[
                                  data["trace_cut_index"][1]:data["trace_cut_index"][number_of_traces_tested - 1]]
    else:
        original_time_short = np.copy(original_time)
        original_position_short = np.copy(data["position"])
        trace_cut_idx = data["trace_cut_index"]

    # res = differential_evolution(func=_minimize,
    #                             bounds=bounds,
    #                             popsize=32,
    #                             args=(
    #                                 data, original_time, signal_interpolated, highcut_position, interpolated_delay,
    #                                 consider_all_traces),
    #                             disp=debug)
    # res = shgo(func=_minimize,
    #           bounds=bounds,
    #           n=128,
    #           args=(
    #               data, original_time, signal_interpolated, highcut_position, interpolated_delay,
    #               consider_all_traces),
    #           options={"disp": debug})
    # res = dual_annealing(func=_minimize,
    #                     bounds=bounds,
    #                     args=(
    #                         data, original_time, signal_interpolated, highcut_position, interpolated_delay,
    #                         consider_all_traces))
    # res = basinhopping(func=_minimize,
    #                   x0=0,
    #                   disp=True,
    #                   minimizer_kwargs={
    #                       "args": (data, original_time, signal_interpolated, highcut_position, interpolated_delay,
    #                                consider_all_traces)})
    start = timer()
    res = direct(func=_minimize,
                 bounds=bounds,
                 len_tol=5e-6,
                 args=(data, original_time_short, original_position_short, trace_cut_idx, signal_interpolated,
                       interpolated_delay))
    # print(res)
    end = timer()
    config.logger.info(f"Global search for compensating phase delay took {EngFormatter('s', places=1)(end - start)}.")
    if debug:
        fig, axs = plot_debug.optimizing_delay(iteration_steps, iteration_delays, iteration_errors)
    return res.x[0]


def get_local_delay(data, original_time, signal_interpolated, consider_all_traces,
                    local_search_startpoint=50, debug=False):
    global iteration_steps
    global iteration_delays
    global iteration_errors
    interpolated_delay = np.linspace(0, 1, data["interpolation_resolution"])
    x0 = [0]
    init_simplex = np.array([0, local_search_startpoint]).reshape(2, 1)
    xatol = 0.1
    iteration_steps = []
    iteration_delays = []
    iteration_errors = []
    original_position_short = None
    original_time_short = None
    trace_cut_idx = None
    number_of_traces_tested = 102
    if not consider_all_traces and data["number_of_traces"] > number_of_traces_tested:
        #  Cut position and signal data to the first 100 traces to accelerate shifting of the arrays
        #  and using less memory
        original_time_short = np.copy(original_time)[
                              data["trace_cut_index"][1]:data["trace_cut_index"][number_of_traces_tested - 1]]
        trace_cut_idx = data["trace_cut_index"][(data["trace_cut_index"] >= data["trace_cut_index"][2]) &
                                                (data["trace_cut_index"] <= data["trace_cut_index"][
                                                    number_of_traces_tested - 2])]
        trace_cut_idx -= data["trace_cut_index"][1]
        original_position_short = np.copy(data["position"])[
                                  data["trace_cut_index"][1]:data["trace_cut_index"][number_of_traces_tested - 1]]
    else:
        original_time_short = np.copy(original_time)
        original_position_short = np.copy(data["position"])
        trace_cut_idx = data["trace_cut_index"]
    res = minimize(_minimize,
                   x0,
                   method="Nelder-Mead",
                   args=(data, original_time_short, original_position_short, trace_cut_idx, signal_interpolated,
                         interpolated_delay),
                   options={"disp": debug,
                            "maxiter": 30,
                            "xatol": xatol,
                            "initial_simplex": init_simplex})
    if debug:
        fig, axs = plot_debug.optimizing_delay(iteration_steps, iteration_delays, iteration_errors)
    return res.x[0]


def _minimize(delay, data, original_time_short, original_position_short, trace_cut_idx, signal_interpolated,
              interpolated_delay):
    new_time_axis = delay * data["dt"] + original_time_short
    new_signal = signal_interpolated(new_time_axis)
    new_signal = (new_signal - np.nanmin(new_signal)) / (np.nanmax(new_signal) - np.nanmin(new_signal))

    new_position = original_position_short
    new_position = (new_position - np.min(new_position)) / (np.max(new_position) - np.min(new_position))

    new_position = new_position[~np.isnan(new_signal)]
    new_signal = new_signal[~np.isnan(new_signal)]

    traces_for_testing = zip(np.split(new_position, trace_cut_idx),
                             np.split(new_signal, trace_cut_idx))

    signal_matrix = np.zeros((data["interpolation_resolution"], len(trace_cut_idx) + 1))
    signal_matrix[:] = np.NaN

    i = 0
    for position, signal in traces_for_testing:
        # Numpy's interpolation method needs sorted, strictly increasing values
        signal = signal[np.argsort(position)]
        position = position[np.argsort(position)]
        # Since it needs to be strictly increasing, keep only values where x is strictly increasing.
        # Ignore any other y value when it has the same x value.
        # signal = np.append(signal[0], signal[1:][(np.diff(position) > 0)])
        # position = np.append(position[0], position[1:][(np.diff(position) > 0)])

        signal = np.interp(interpolated_delay, position, signal)
        signal_matrix[:, i] = signal
        i += 1

    # config.logger.info(f"Delay:\t{delay[0]:.3f}\tError:\t{np.sum(np.nanstd(signal_matrix, axis=1))}")
    current_cost = np.sum(np.nanstd(signal_matrix, axis=1))
    _housekeeping_optimizer(delay, current_cost)
    return current_cost


def shift_position(data, original_time, signal_interpolated):
    config.logger.info(
        f"Found optimal delay_value {EngFormatter('Sa', places=3)(data['delay_value'])}, "
        f"corresponding to a time delay of {EngFormatter('s')(data['delay_value'] * data['dt'])}.")
    new_time_axis = data["delay_value"] * data["dt"] + np.copy(original_time)
    data["signal"] = signal_interpolated(new_time_axis)
    data["position"] = data["position"][~np.isnan(data["signal"])]
    data["signal"] = data["signal"][~np.isnan(data["signal"])]


def butter_filter(data, fs, lowcut=None, highcut=None, order=5):
    sos = _butter_coeff(fs, lowcut, highcut, order=order)
    y = sosfiltfilt(sos, data, padtype=None)
    return y


def _butter_coeff(fs, lowcut=None, highcut=None, order=None):
    """Create coefficients for a butterworth filter."""
    nyq = 0.5 * fs
    if highcut is not None and highcut > nyq:
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
