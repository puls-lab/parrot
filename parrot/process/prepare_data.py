import numpy as np
from scipy.signal import sosfiltfilt, butter, find_peaks, correlate, correlation_lags
from scipy.optimize import minimize
# TODO: Remove matplotlib later (but not EngFormatter)
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from numba import njit
# TODO: Delete time later
import time


class PrepareData:
    def __init__(self,
                 raw_data,
                 recording_type=None,
                 scale=None,
                 max_THz_frequency=50e12,
                 delay_value=None,
                 filter_position=True,
                 lowcut_position=None,
                 highcut_position=100,
                 filter_signal=True,
                 lowcut_signal=1,
                 highcut_signal=None,
                 debug=False):
        # scale:        Scale between [V] of position data and light time [s]
        #               For example for the APE 50 ps shaker it would be scale=50e-12/20 (50 ps for +-10V)
        # delay-value:  Delay between recorded position data and signal data (due to i.e. different bandwidth)
        # position/signal: Copy raw-data to pre-processed data
        self.data = {"scale": scale,
                     "delay_value": delay_value,
                     "position": raw_data["position"],
                     "signal": raw_data["signal"]}
        self.recording_type = recording_type
        self.debug = debug
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
        self.dt = (raw_data["time"][-1] - raw_data["time"][0]) / (len(raw_data["time"]) - 1)
        self.data["number_of_traces"] = None
        self.data["interpolation_resolution"] = None
        self.data["trace_cut_index"] = None

    def run(self):
        if self.filter_position:
            self.data["position"] = self.butter_filter(self.data["position"],
                                                       1 / self.dt,
                                                       lowcut=self.lowcut_position,
                                                       highcut=self.highcut_position)
        # Calculate the total record length in THz time, afterward we can select the correct interpolation resolution
        self.data["thz_recording_length"] = self.data["scale"] * (
                np.max(self.data["position"]) - np.min(self.data["position"]))
        self.data["thz_start_offset"] = self.data["scale"] * np.min(self.data["position"])
        if self.filter_signal:
            if self.highcut_signal is None:
                self.highcut_signal = self.max_THz_frequency
            self.data["signal"] = self.butter_filter(self.data["signal"],
                                                     1 / self.dt,
                                                     lowcut=self.lowcut_signal,
                                                     highcut=self.highcut_signal)
            self.resample_data()
        if self.recording_type == "single_cycle":
            if np.argmin(self.data["position"]) - np.argmax(self.data["position"]) < 0:
                # Either first minimum, then maximum
                idx = np.array([np.argmin(self.data["position"]), np.argmax(self.data["position"])])
            else:
                # Otherwise, first maximum, then minimum
                idx = np.array([np.argmax(self.data["position"]), np.argmin(self.data["position"])])
        else:
            # Get the peaks of the sinusoid (or similar) of the position data, then we know the number of traces
            self.data["trace_cut_index"] = self.get_multiple_index()
        self.cut_incomplete_traces()
        if self.debug:
            fig, ax = plt.subplots()
            ax.plot(self.data["position"])
            [ax.axvline(x, color="black", alpha=0.8) for x in self.data["trace_cut_index"]]
            ax.set_xlabel("Time sample")
            ax.set_ylabel("Voltage")
            ax.grid(True)
        for exponent in range(6, 20):
            if 0.5 * (2 ** exponent / self.data["thz_recording_length"]) > self.max_THz_frequency:
                self.data["interpolation_resolution"] = 2 ** exponent
                if self.debug:
                    print("INFO: Found interpolation resolution to have more than "
                          f"{EngFormatter('Hz', places=1)(self.max_THz_frequency)}: 2 ** {exponent} = "
                          f"{2 ** exponent} points")
                break
        if self.data["interpolation_resolution"] is None:
            raise ValueError("Could not find a proper interpolation resolution between 2**6 and 2**20."
                             "Did you select the right scale [ps/V] and the right max_THz_frequency?")
        # If we recorded multiple forward/backward traces, we can calculate the delay between position and signal.
        if self.recording_type == "multi_cycle":
            # Get timedelay between position and signal
            if self.debug:
                fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
                split_pos = np.split(self.data["position"], self.data["trace_cut_index"])
                split_sig = np.split(self.data["signal"], self.data["trace_cut_index"])
                for i in range(1, 10):
                    if i % 2:
                        ax[0].plot(self.data["scale"] * split_pos[i],
                                   split_sig[i],
                                   color="tab:blue", alpha=0.8)
                    else:
                        ax[0].plot(self.data["scale"] * split_pos[i],
                                   split_sig[i],
                                   color="tab:orange", alpha=0.8)
                ax[0].xaxis.set_major_formatter(EngFormatter(unit='s'))
                ax[0].yaxis.set_major_formatter(EngFormatter(unit='V'))
                ax[0].set_title('Signal vs. Delay without time delay compensation')
            if self.data["delay_value"] is None:
                self.data["delay_value"] = self.get_delay()
            self.shift_position(self.data["delay_value"])
            if self.debug:
                split_pos = np.split(self.data["position"], self.data["trace_cut_index"])
                split_sig = np.split(self.data["signal"], self.data["trace_cut_index"])
                for i in range(1, 10):
                    if i % 2:
                        ax[1].plot(self.data["scale"] * split_pos[i],
                                   split_sig[i],
                                   color="tab:blue", alpha=0.8)
                    else:
                        ax[1].plot(self.data["scale"] * split_pos[i],
                                   split_sig[i],
                                   color="tab:orange", alpha=0.8)
                ax[1].xaxis.set_major_formatter(EngFormatter(unit='s'))
                ax[1].yaxis.set_major_formatter(EngFormatter(unit='V'))
                delay_amount = self.data["delay_value"] * self.dt
                ax[1].set_title(
                    f'Signal vs. Delay with {self.data["delay_value"]} samples ' +
                    f'({EngFormatter("s", places=1)(delay_amount)}) time delay compensation')
                plt.tight_layout()
        return self.data

    def resample_data(self):
        """This is a little bit tricky, since we have the sampling time in lab time but not in "light time" [ps].
        The self.max_THz_frequency is defined in the time frame of the THz sample.

        The max. slope of the position data vs. lab time is the smallest max. THz frequency
        """
        # TODO: Needs to be checked
        max_native_frequency = 1 / (np.max(np.gradient(self.data["position"], self.dt)) * self.data[
            "scale"] * self.dt)  # [V/s] * [ps/V] --> scaling factor
        factor = np.int64(np.floor(max_native_frequency / self.max_THz_frequency))
        current_time = np.arange(0, len(self.data["position"]) * self.dt, self.dt)
        if factor < 1:
            factor = 1
        new_dt = factor * self.dt
        new_time = np.arange(0, len(self.data["position"]) * self.dt, new_dt)
        if self.debug:
            print(
                f"INFO: Current time sample: {self.dt}s per sample. New time sample: {new_dt}s per sample.")
        self.data["position"] = np.interp(new_time, current_time, self.data["position"])
        self.data["signal"] = np.interp(new_time, current_time, self.data["signal"])
        self.dt = new_dt

    def get_multiple_index(self):
        position = self.data["position"] - np.mean(self.data["position"])
        if self.filter_position and self.highcut_position is not None:
            original_max_freq = 1 / self.dt
            new_max_freq = 10 * self.highcut_position
            reduction_factor = int(np.round(original_max_freq / new_max_freq))
            signal_fft = np.abs(np.fft.rfft(np.abs(position[::reduction_factor])))
            freq = np.fft.rfftfreq(len(position[::reduction_factor]), reduction_factor * self.dt)
            # Excluding the zero frequency "peak", which is related to offset
            guess_freq = np.abs(freq[np.argmax(signal_fft[1:]) + 1])
        else:
            start = time.time()
            signal_fft = np.abs(np.fft.rfft(np.abs(position)))
            freq = np.fft.rfftfreq(len(position), self.dt)
            guess_freq = np.abs(freq[np.argmax(signal_fft[1:]) + 1])
            end = time.time()
            print(f"INFO: Taking rFFT over complete position array, taking {EngFormatter('s', places=1)(end - start)}."
                  "If you specify filter_position=True and a reasonable highcut_position (in [Hz]), "
                  "you can accelerate this process alot.")

        idx, _ = find_peaks(np.abs(position),
                            height=0.8 * np.max(np.abs(position)),
                            distance=round(0.9 * (1 / guess_freq) / self.dt))
        return idx

    def cut_incomplete_traces(self):
        """Cut any incomplete trace from the array before the first delay peak or after the last delay peak"""
        self.data["position"] = self.data["position"][self.data["trace_cut_index"][0]:self.data["trace_cut_index"][-1]]
        self.data["signal"] = self.data["signal"][self.data["trace_cut_index"][0]:self.data["trace_cut_index"][-1]]
        self.data["trace_cut_index"] = self.data["trace_cut_index"][1:-1] - self.data["trace_cut_index"][0]
        self.data["number_of_traces"] = len(self.data["trace_cut_index"]) + 1

    def get_delay(self):
        x0 = [0]
        init_simplex = np.array([0, 10]).reshape(2, 1)
        xatol = 0.5
        fatol = 0.5
        if self.debug:
            print("INFO: Optimizing delay between position and signal array to align forward and backward THz traces.")
        res = minimize(self.minimize_delay,
                       x0,
                       method="Nelder-Mead",
                       options={"disp": False,
                                "maxiter": 30,
                                "fatol": fatol,
                                "xatol": xatol,
                                "initial_simplex": init_simplex})
        if self.debug:
            print(f"INFO: A delay of {int(np.round(res.x[0]))} minimizes the error "
                  "and aligns forward and backward traces.")
        return int(np.round(res.x[0]))

    def minimize_delay(self, delay):
        delay = int(np.round(delay))
        pos = np.copy(self.data["position"])
        sig = np.copy(self.data["signal"])
        if delay < 0:
            sig = np.roll(sig, delay)
            pos[delay:] = np.nan
            sig[delay:] = np.nan
        else:
            sig = np.roll(sig, delay)
            pos[:delay] = np.nan
            sig[:delay] = np.nan
        pos = pos[~np.isnan(pos)]
        sig = sig[~np.isnan(sig)]
        pos_list = np.split(pos, self.data["trace_cut_index"])
        sig_list = np.split(sig, self.data["trace_cut_index"])

        sorted_first_signal = sig_list[0][pos_list[0].argsort()]
        first_trace = sorted_first_signal - np.mean(sorted_first_signal)

        all_lags_squared = 0
        for pos_trace, sig_trace in zip(pos_list, sig_list):
            current_trace = sig_trace[pos_trace.argsort()]
            correlation = correlate(first_trace,
                                    current_trace - np.mean(current_trace), mode="same")
            lags = correlation_lags(first_trace.size, current_trace.size, mode="same")
            lag = lags[np.argmax(correlation)]
            all_lags_squared += lag ** 2
        if self.debug:
            print(f"Time sample delay:\t{int(delay)}\tError^2:\t{all_lags_squared:.1e}")
        return all_lags_squared

    def shift_position(self, delay_value):
        self.data["signal"] = np.roll(self.data["signal"], delay_value)
        if delay_value < 0:
            self.data["position"][delay_value:] = np.nan
            self.data["signal"][delay_value:] = np.nan
        else:
            self.data["position"][:delay_value] = np.nan
            self.data["signal"][:delay_value] = np.nan
        self.data["position"] = self.data["position"][~np.isnan(self.data["position"])]
        self.data["signal"] = self.data["signal"][~np.isnan(self.data["signal"])]

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
