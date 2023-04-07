import numpy as np
from scipy.signal import sosfiltfilt, butter, find_peaks
from scipy.optimize import minimize
from matplotlib.ticker import EngFormatter
from . import Process


class PrepareData(Process):
    def __init__(self,
                 raw_data,
                 scale=None,
                 max_THz_frequency=100e12,
                 delay_value=None,
                 filter_position=True,
                 lowcut_position=None,
                 highcut_position=100,
                 filter_signal=False,
                 lowcut_signal=1,
                 highcut_signal=None):
        super().__init__()
        # scale: Scale between [V] of position data and light time [s]
        #        For example for the APE 50 ps shaker it would be scale=50e-12/20 (50 ps for +-10V)
        # delay-Value: Delay between recorded position data and signal data (due to i.e. different bandwidth)
        # position/signal: Copy raw-data to pre-processed data
        self.data = {"scale": scale,
                     "delay_value": delay_value,
                     "position": raw_data["position"],
                     "signal": raw_data["signal"]}
        # Maximum THz frequency, which can be later displayed with interpolated data.
        # Large values mean many interpolation points, this can fill up RAM quickly when data contains 1000s of traces.
        self.max_THz_frequency = max_THz_frequency
        # Possible lowcut/highcut filtering of position and signal
        self.filter_position = filter_position
        self.lowcut_position = lowcut_position
        self.highcut_position = highcut_position
        self.filter_signal = filter_signal
        self.lowcut_signal = lowcut_signal
        self.highcut_signal = highcut_signal
        # Timestep in lab time
        self.dt = (raw_data["time"].iloc[-1] - raw_data["time"].iloc[0]) / (len(raw_data["time"]) - 1)
        self.data["number_of_traces"] = None
        self.data["interpolation_resolution"] = None
        self.trace_idx = None
        self.run()

    def run(self):
        if self.filter_position:
            self.data["position"] = self.butter_filter(self.data["position"],
                                                       1 / self.dt,
                                                       lowcut=self.lowcut_position,
                                                       highcut=self.highcut_position)
        if self.filter_signal:
            self.data["signal"] = self.butter_filter(self.data["signal"],
                                                     1 / self.dt,
                                                     lowcut=self.lowcut_signal,
                                                     highcut=self.highcut_signal)
        if self.recording_type == "single_cycle":
            if np.argmin(self.data["position"]) - np.argmax(self.data["position"]) < 0:
                # Either first minimum, then maximum
                idx = np.array([np.argmin(self.data["position"]), np.argmax(self.data["position"])])
            else:
                # Otherwise, first maximum, then minimum
                idx = np.array([np.argmax(self.data["position"]), np.argmin(self.data["position"])])
        else:
            # Get the peaks of the sinusoid (or similar) of the position data, then we know the number of traces
            idx = self.get_multiple_index(self.data["position"])
        self.data["number_of_traces"] = len(idx) - 1
        # Calculate the total record length in THz time, afterwards we can select the correct interpolation resolution
        thz_recording_length = self.data["scale"] * np.max(self.data["position"]) - np.min(self.data["position"])
        for exponent in range(6, 20):
            if 0.5 * (2 ** exponent / thz_recording_length) > self.max_THz_frequency:
                self.data["interpolation_resolution"] = 2 ** exponent
                if self.debug:
                    print(f"Found interpolation resolution to have more than {self.max_THz_frequency}: 2 ** {exponent}="
                          f"{2 ** exponent} points")
        if self.data["interpolation_resolution"] is None:
            raise ValueError("Could not find a proper interpolation resolution between 2**6 and 2**20."
                             "Did you select the right scale [ps/V] and the right max_THz_frequency?")
        self.trace_idx = np.zeros(len(self.data["position"]))
        self.trace_idx[idx] = 1
        # Contains the index of the trace
        self.trace_idx = np.cumsum(self.trace_idx).astype(int)
        self.cut_incomplete_traces()
        # If we recorded multiple forward/backward traces, we can calculate the delay between position and signal.
        if self.recording_type == "multi_cycle":
            # Get timedelay between position and signal
            if self.data["delay_value"] is None:
                self.data["delay_value"] = self.get_delay()
            self.shift_position(self.data["delay_value"])
        return self.data

    def get_multiple_index(self, position):
        position = position - np.mean(position)

        ff = np.fft.fftfreq(len(position), self.dt)
        fyy = abs(np.fft.fft(np.abs(position)))
        guess_freq = abs(ff[np.argmax(fyy[1:]) + 1])  # excluding the zero frequency "peak", which is related to offset

        idx, _ = find_peaks(np.abs(position),
                            height=0.8 * np.max(np.abs(position)),
                            distance=round(0.9 * (1 / guess_freq) / self.dt))
        return idx

    def cut_incomplete_traces(self):
        """Cut any incomplete trace from the array before the first delay peak or after the last delay peak"""
        mask = (self.trace_idx == 0) | (self.trace_idx == np.max(self.trace_idx))
        self.data["position"] = self.data["position"][~mask]
        self.data["signal"] = self.data["signal"][~mask]
        self.trace_idx = self.trace_idx[~mask]

    def get_delay(self):
        x0 = [0]
        init_simplex = np.array([0, 10]).reshape(2, 1)
        xatol = 0.5
        fatol = 0.5
        res = minimize(self._minimize, x0, method="Nelder-Mead", options={"disp": False,
                                                                          "maxiter": 30,
                                                                          "fatol": fatol,
                                                                          "xatol": xatol,
                                                                          "initial_simplex": init_simplex})
        return int(res.x[0])

    def _minimize(self, delay):
        # start = timer()
        delay = int(delay)
        pos = np.copy(self.data["position"])
        sig = np.copy(self.data["signal"])
        idx = np.copy(self.trace_idx)
        if delay < 0:
            pos = np.roll(pos, delay)
            pos[delay:] = np.nan
            sig[delay:] = np.nan
            idx[delay:] = -1  # int does not know float nan (only works for selected arraylength > 1)
        else:
            pos = np.roll(pos, delay)
            pos[:delay] = np.nan
            sig[:delay] = np.nan
            idx[:delay] = -1
        pos = pos[~np.isnan(pos)]
        sig = sig[~np.isnan(sig)]
        idx = idx[idx > 0]
        peak_loc = np.zeros(len(np.unique(idx)))
        for i, current_idx in enumerate(np.unique(idx)):
            current_pos = pos[idx == current_idx]
            current_sig = sig[idx == current_idx]
            peak_loc[i] = current_pos[np.argmax(current_sig)]
        # df = pd.DataFrame(data={"position": pos, "signal": sig, "trace": idx})
        # peak_loc = df.loc[df.groupby(["trace"])["signal"].idxmax(), "position"]
        # Subtract avg. delay position at signal peak between forward/backward traces
        diff = np.mean(peak_loc[::2]) - np.mean(peak_loc[1::2])
        if self.debug:
            print(f"Delay:\t{int(delay)}\tDiff:\t{diff:.3f}\tDiff^2:\t{diff ** 2}")
            # end = timer()
            # print(f"{(end - start):.3f} s")
        return diff ** 2

    def shift_position(self, delay_value):
        self.data["position"] = np.roll(self.data["position"], delay_value)
        if delay_value < 0:
            self.data["position"][delay_value:] = np.nan
            self.data["signal"][delay_value:] = np.nan
            self.trace_idx[delay_value:] = -1
        else:
            self.data["position"][:delay_value] = np.nan
            self.data["signal"][:delay_value] = np.nan
            self.trace_idx[:delay_value] = -1
        self.data["position"] = self.data["position"][~np.isnan(self.data["position"])]
        self.data["signal"] = self.data["signal"][~np.isnan(self.data["signal"])]
        # int doesn't understand np.nan, so it takes -MAX_INT value
        self.trace_idx = self.trace_idx[self.trace_idx > 0]

    def butter_filter(self, data, fs, lowcut=None, highcut=None, order=5):
        sos = self._butter_coeff(fs, lowcut, highcut, order=order)
        y = sosfiltfilt(sos, data, padtype=None)
        return y

    @staticmethod
    def _butter_coeff(fs, lowcut=None, highcut=None, order=None):
        """Create coefficients for a butterworth filter."""
        nyq = 0.5 * fs
        if highcut > nyq:
            print(
                f"INFO: {EngFormatter('Hz')(highcut)} > Nyquist-frequency ({EngFormatter('Hz')(nyq)}), "
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
