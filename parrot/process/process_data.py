import numpy as np

# Own functions of parrot
# from .prepare_data import PrepareData
from ..process import prepare_data
from ..plot import plot
from ..config import config

"""
import logging
# Set-up logger
logger = logging.getLogger(__name__)
logger.handlers.clear()

formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
"""


def _calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft


def _process_all_traces(matrix, total_position, total_signal, trace_idx, interpolated_delay):
    """Splits the data according to detected extrema in the positional data and makes an equidistant, normalized
    interpoaltion."""
    pos_list = np.split(total_position, trace_idx)
    sig_list = np.split(total_signal, trace_idx)
    i = 0
    for position, signal in zip(pos_list, sig_list):
        # Numpy's interpolation method needs sorted, strictly increasing values
        signal = signal[np.argsort(position)]
        position = position[np.argsort(position)]
        # Since it needs to be strictly increasing, keep only values where x is strictly increasing.
        # Ignore any other y value when it has the same x value.
        signal = np.append(signal[0], signal[1:][(np.diff(position) > 0)])
        position = np.append(position[0], position[1:][(np.diff(position) > 0)])
        matrix[:, i] = np.interp(interpolated_delay, position, signal)
        i += 1
    return matrix


def normalize_and_process(data):
    """Normalize the position signal (i.e. a voltage signal from a shaker) between 0 and 1."""
    position = (data["position"] - np.min(data["position"])) / (np.max(data["position"]) - np.min(data["position"]))
    data["interpolated_position"] = np.linspace(0, 1, data["interpolation_resolution"])
    # Create matrix, containing each single, interpolated THz trace
    matrix = np.zeros((data["interpolation_resolution"], data["number_of_traces"]))
    config.logger.info(f"Creating matrix with {matrix.shape}. Starting interpolation for all traces...")
    matrix = _process_all_traces(matrix,
                                 position,
                                 data["signal"],
                                 data["trace_cut_index"],
                                 data["interpolated_position"])
    data["light_time"] = data["thz_recording_length"] * data["interpolated_position"] + data["thz_start_offset"]
    data["single_traces"] = matrix
    data["average"] = {"time_domain": np.mean(matrix, axis=1)}
    return data


def thz_and_dark(light, dark, recording_type="multi_cycle", **kwargs):
    raw_data = {}
    for mode, data in zip(["light", "dark"], [light, dark]):
        raw_data[mode] = {"time": data["time"],
                          "position": data["position"],
                          "signal": data["signal"]}
    data = {"light": prepare_data.run(raw_data["light"], recording_type=recording_type, dataset_name="light", **kwargs),
            "dark": {}}
    if 'delay_value' in kwargs:
        data["dark"] = prepare_data.run(raw_data["dark"], recording_type=recording_type, dataset_name="dark", **kwargs)
    else:
        data["dark"] = prepare_data.run(raw_data["dark"],
                                        recording_type=recording_type,
                                        dataset_name="dark",
                                        delay_value=data["light"]["delay_value"],
                                        **kwargs)
    for mode in ["light", "dark"]:
        data[mode] = normalize_and_process(data[mode])
        if config.get_debug() and mode == "light":
            fig, ax = plot.debug_analysis_amplitude_jitter(data["light"])
        frequency, signal_fft = _calc_fft(data[mode]["light_time"], data[mode]["average"]["time_domain"])
        data[mode]["frequency"] = frequency
        data[mode]["average"]["frequency_domain"] = signal_fft
    data["applied_functions"] = []
    return data


def thz_and_two_darks(light, dark1, dark2, recording_type="multi_cycle", **kwargs):
    raw_data = {}
    for mode, data in zip(["light", "dark1", "dark2"], [light, dark1, dark2]):
        raw_data[mode] = {"time": data["time"],
                          "position": data["position"],
                          "signal": data["signal"]}
    data = {"light": prepare_data.run(raw_data["light"], recording_type=recording_type, dataset_name="light", **kwargs),
            "dark1": {},
            "dark2": {}}
    if 'delay_value' in kwargs:
        data["dark1"] = prepare_data.run(raw_data["dark1"], recording_type=recording_type, dataset_name="dark1",
                                         **kwargs)
        data["dark2"] = prepare_data.run(raw_data["dark2"], recording_type=recording_type, dataset_name="dark2",
                                         **kwargs)
    else:
        data["dark1"] = prepare_data.run(raw_data["dark1"],
                                         recording_type=recording_type,
                                         dataset_name="dark1",
                                         delay_value=data["light"]["delay_value"],
                                         **kwargs)
        data["dark2"] = prepare_data.run(raw_data["dark2"],
                                         recording_type=recording_type,
                                         dataset_name="dark2",
                                         delay_value=data["light"]["delay_value"],
                                         **kwargs)
    for mode in ["light", "dark1", "dark2"]:
        data[mode] = normalize_and_process(data[mode])
        if config.get_debug() and mode == "light":
            fig, ax = plot.debug_analysis_amplitude_jitter(data["light"])
        frequency, signal_fft = _calc_fft(data[mode]["light_time"], data[mode]["average"]["time_domain"])
        data[mode]["frequency"] = frequency
        data[mode]["average"]["frequency_domain"] = signal_fft
    data["applied_functions"] = []
    return data


def thz_only(light, recording_type="multi_cycle", **kwargs):
    raw_data = {"light": {"time": light["time"],
                          "position": light["position"],
                          "signal": light["signal"]}}
    data = {"light": prepare_data.run(raw_data["light"], recording_type=recording_type, dataset_name="light", **kwargs)}
    data["light"] = normalize_and_process(data["light"])
    if config.get_debug():
        fig, ax = plot.debug_analysis_amplitude_jitter(data["light"])
    frequency, signal_fft = _calc_fft(data["light"]["light_time"], data["light"]["average"]["time_domain"])
    data["light"]["frequency"] = frequency
    data["light"]["average"]["frequency_domain"] = signal_fft
    data["applied_functions"] = []
    return data


def dark_only(dark, recording_type="multi_cycle", **kwargs):
    raw_data = {"dark": {"time": dark["time"],
                         "position": dark["position"],
                         "signal": dark["signal"]}}
    data = {"dark": prepare_data.run(raw_data["dark"], recording_type=recording_type, dataset_name="dark", **kwargs)}
    data["dark"] = normalize_and_process(data["dark"])
    frequency, signal_fft = _calc_fft(data["dark"]["light_time"], data["dark"]["average"]["time_domain"])
    data["dark"]["frequency"] = frequency
    data["dark"]["average"]["frequency_domain"] = signal_fft
    data["applied_functions"] = []
    return data
