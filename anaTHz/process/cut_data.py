import numpy as np
from numba import njit, prange


@njit(parallel=True)
def parallel_processing(matrix, total_position, total_signal, trace_idx, interpolated_delay):
    pos_list = np.split(total_position, trace_idx)
    sig_list = np.split(total_signal, trace_idx)
    for i in prange(len(pos_list)):
        position = pos_list[i]
        signal = sig_list[i]
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
    return matrix


class CutData:
    def __init__(self, data, debug=True):
        # super().__init__()
        self.data = data
        self.data["interpolated_position"] = np.linspace(0, 1, self.data["interpolation_resolution"])
        self.debug = debug
        self.run()

    def run(self):
        """Normalize the position signal (i.e. a voltage signal from a shaker) between 0 and 1."""
        position = (self.data["position"] - np.min(self.data["position"])) / (np.max(self.data["position"]) -
                                                                              np.min(self.data["position"]))
        # Create matrix, containing each single, interpolated THz trace
        matrix = np.zeros((self.data["interpolation_resolution"], self.data["number_of_traces"]))
        if self.debug:
            print(f"Creating matrix with {matrix.shape}. Starting parallel-processing...")
        matrix = parallel_processing(matrix,
                                     position,
                                     self.data["signal"],
                                     self.data["trace_cut_index"],
                                     self.data["interpolated_position"])
        self.data["interpolated_position"] *= self.data["scale"]
        self.data["single_traces"] = matrix
        return self.data
