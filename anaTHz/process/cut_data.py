import numpy as np
from numba import njit, prange
# Own library
from . import Process


@njit(parallel=True)
def parallel_processing(matrix, total_position, total_signal, trace_idx, interpolated_delay):
    for i in prange(np.unique(trace_idx)):
        position = total_position[trace_idx == i]
        signal = total_signal[trace_idx == i]
        signal = signal[np.argsort(position)]
        position = position[np.argsort(position)]
        # Numpy interp needs x to be stricly increasing:
        if np.all(np.diff(position) > 0):
            signal = np.interp(interpolated_delay, position, signal)
        else:
            raise ValueError("Position data does not increase strictly, which is problematic for numpy interpolation.")
        # Old version based on scipy interp1d
        # f_interp = interp1d(position,
        #                    signal,
        #                    bounds_error=False,
        #                    fill_value=(0, 0))
        # signal = f_interp(interpolated_delay)
        matrix[:, i - np.min(trace_idx)] = signal
    return matrix


class CutData(Process):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.data["interpolated_position"] = np.linspace(0, 1, self.data["interpolation_resolution"])
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
                                     self.data["trace_idx"],
                                     self.data["interpolated_position"])
        self.data["interpolated_position"] *= self.data["scale"]
        self.data["all_traces"] = matrix
        return self.data
