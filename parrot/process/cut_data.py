import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def process_all_traces(matrix, total_position, total_signal, trace_idx, interpolated_delay):
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
        # Old version based on scipy interp1d
        # f_interp = interp1d(position,
        #                    signal,
        #                    bounds_error=False,
        #                    fill_value=(0, 0))
        # signal = f_interp(interpolated_delay)
        matrix[:, i] = np.interp(interpolated_delay, position, signal)
        i += 1
    return matrix


class CutData:
    def __init__(self, data, debug=True):
        # super().__init__()
        self.data = data
        self.data["interpolated_position"] = np.linspace(0, 1, self.data["interpolation_resolution"])
        self.debug = debug
        if not debug:
            logger.setLevel(logging.WARNING)

    def run(self):
        """Normalize the position signal (i.e. a voltage signal from a shaker) between 0 and 1."""
        position = (self.data["position"] - np.min(self.data["position"])) / (np.max(self.data["position"]) -
                                                                              np.min(self.data["position"]))
        # Create matrix, containing each single, interpolated THz trace
        matrix = np.zeros((self.data["interpolation_resolution"], self.data["number_of_traces"]))
        logger.info(f"Creating matrix with {matrix.shape}. Starting interpolation for all traces...")
        matrix = process_all_traces(matrix,
                                    position,
                                    self.data["signal"],
                                    self.data["trace_cut_index"],
                                    self.data["interpolated_position"])
        self.data["light_time"] = self.data["thz_recording_length"] * self.data["interpolated_position"] + \
                                  self.data["thz_start_offset"]
        self.data["single_traces"] = matrix
        self.data["average"] = {"time_domain": np.mean(matrix, axis=1)}
        return self.data
