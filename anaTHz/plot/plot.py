import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from numba import njit


@njit()
def calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft


@njit()
def cumulated_mean_fft(data):
    dt = (data["light_time"][-1] - data["light_time"][0]) / (len(data["light_time"]) - 1)
    frequency = np.fft.rfftfreq(len(data["light_time"]), dt)
    # Cumulative mean of all single traces
    # TODO: Need to check if this is correct
    matrix = np.cumsum(data["single_traces"], axis=0).T / np.arange(1, data["number_of_traces"] + 1)
    matrix = np.fft.rfft(matrix, axis=1)
    return frequency, matrix


class Plot:
    def __init__(self,
                 recording_type=None,
                 min_THz_frequency=0.1e12,
                 max_THz_frequency=5e12,
                 plot_water_absorption=True,
                 n_largest_water_absorptions=30):
        self.start_bandwidth = None
        self.stop_bandwidth = None
        self.bandwidth = None
        self.recording_type = recording_type
        self.min_THz_frequency = min_THz_frequency  # In [Hz]
        self.max_THz_frequency = max_THz_frequency  # In [Hz]
        self.noise_floor_start_fft = noise_floor_start_fft  # In [Hz]
        # True or False to plot water absorption lines in frequency plot
        self.plot_water_absorption = plot_water_absorption
        # Reduce the amount of water absorption lines to the strongest n to not clutter the axis
        self.n_largest_water_absorptions = n_largest_water_absorptions

        self.time = None
        self.frequency = None
        self.filter_frequency = None

    def run(self, data):
        self.filter_frequency = (df["frequency"] > self.min_THz_frequency) & (df["frequency"] < self.max_THz_frequency)
        self.frequency = df.loc[self.filter_frequency, "frequency"]

        if self.recording_type == "single_cycle":
            fig, ax = self.plot_single_cycle(data)
        if self.recording_type == "multi_cycle":
            fig, ax = self.plot_multi_cycle(data)
        return fig, ax

    def plot_full_multi_cycle(self, data, snr_timedomain=True, water_absorption_lines=True):
        fig, ax = plt.subplots(nrows=1, ncols=3)
        # First subplot, time domain
        try:
            std_traces = np.std(data["dark"]["single_traces"], axis=1)  # TODO: Check if this is columns
            max_traces = np.max(data["dark"]["single_traces"], axis=1)
            min_traces = np.min(data["dark"]["single_traces"], axis=1)
            ax[0].fill_between(data["dark"]["light_time"],
                               min_traces,
                               max_traces,
                               color="black",
                               alpha=0.3,
                               label="Min/Max of dark traces")
            ax[0].fill_between(data["dark"]["light_time"],
                               data["dark"]["average"]["time_domain"] - std_traces,
                               data["dark"]["average"]["time_domain"] + std_traces,
                               color="black",
                               alpha=0.7,
                               label="Standard deviation of dark traces")
            ax[0].plot(data["dark"]["light_time"],
                       data["dark"]["average"]["time_domain"],
                       color="black",
                       alpha=0.8,
                       label=f"Average of {data['dark']['number_of_traces']} dark traces")
        except KeyError:
            pass
        std_traces = np.std(data["light"]["single_traces"], axis=1)  # TODO: Check if this is columns
        max_traces = np.max(data["light"]["single_traces"], axis=1)
        min_traces = np.min(data["light"]["single_traces"], axis=1)
        ax[0].fill_between(data["light"]["light_time"],
                           min_traces,
                           max_traces,
                           color="#402e32",
                           alpha=0.3,
                           label="Min/Max of THz traces")
        ax[0].fill_between(data["light"]["light_time"],
                           data["light"]["average"]["time_domain"] - std_traces,
                           data["light"]["average"]["time_domain"] + std_traces,
                           color="#6b443b",
                           alpha=0.7,
                           label="Standard deviation of THz traces")
        ax[0].plot(data["light"]["light_time"],
                   data["light"]["average"]["time_domain"],
                   color="tab:orange",
                   alpha=0.8,
                   label=f"Average of {data['number_of_traces']} traces")

        # If the data got artificially extended with zeros in timedomain,
        # we want to limit the x-axis in timedomain to just zoom on the data.
        # The data is not cutted off, you can still pan the axis window,
        # but its outside the selected range just zero.
        data_start = data["light"]["average"]["time_domain"].nonzero()[0][0]
        data_stop = data["light"]["average"]["time_domain"].nonzero()[0][-1]
        ax[0].set_xlim([data["light"]["light_time"][data_start], data["light"]["light_time"][data_stop]])
        if snr_timedomain:
            # Filter all zeros in array (from windowing, extra padding, etc.) since we cannot divide by 0
            filter_zeros = (std_traces == 0)
            # Since the filter is True, when there is zero, we need to use the opposite of that.
            snr_timedomain_max = np.nanmax(np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) /
                                           std_traces[~filter_zeros])
            snr_timedomain = ax[0, 0].twinx()
            snr_timedomain.plot(data["light"]["light_time"],
                                np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) / std_traces[
                                    ~filter_zeros],
                                color="tab:green",
                                alpha=0.4)
            snr_timedomain.set_ylabel("SNR")
            ax[0].scatter([], [], c="tab:green", label=f'SNR, timedomain, max: {int(snr_timedomain_max)}')
        ax[0].legend(loc='upper right')
        ax[0].xaxis.set_major_formatter(EngFormatter(unit='s'))
        ax[0].set_xlabel("Time")
        ax[0].yaxis.set_major_formatter(EngFormatter(unit='V'))

        # Second subplot, frequency-domain
        self.extract_bandwidth(data)
        # Prepare data, for an accumulated mean fo all single-traces calculate the FFT and dynamic range
        frequency_dark, matrix_dark_fft = cumulated_mean_fft(data["dark"])
        frequency_light, matrix_light_fft = cumulated_mean_fft(data["light"])
        filter_frequency = (frequency_dark >= self.start_bandwidth) & (frequency_dark <= self.stop_bandwidth)
        dark_norm = np.mean(np.abs(matrix_dark_fft[-1, :][filter_frequency]) ** 2)
        filter_frequency = (frequency_dark >= self.min_THz_frequency) & (frequency_dark <= self.max_THz_frequency)
        frequency, signal_fft = frequency_dark[filter_frequency], matrix_dark_fft[-1, :][filter_frequency]
        ax[1].plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
                   color="black",
                   alpha=0.8,
                   label=f"{data['dark']['number_of_traces']} dark averages")
        ax[1].axvline(self.start_bandwidth, "--", color="black", alpha=0.5)
        ax[1].axvline(self.stop_bandwidth, "--", color="black", alpha=0.5)
        # To not clutter the plot with too many curves, just plot an accumulated average of multiples of curves,
        # means:
        # If the dataframe contains 573 single THz traces,
        # then plot the average (and FFT) of 1, 10, 100 and all 573 traces.
        trace_range = np.logspace(0,
                                  np.floor(np.log10(data["light"]['number_of_traces'])),
                                  num=1 + int(np.floor(np.log10(data["light"]["number_of_traces"])))
                                  ).astype(int)

        # Make the curves which contain less averaged spectra more transparent
        alpha_values = np.arange(0.8 - 0.15 * len(trace_range), 1, 0.15)
        for j, i in enumerate(trace_range):
            # Select a temporary subset of the dataframe, first dark
            filter_frequency = (frequency_dark >= self.start_bandwidth) & (frequency_dark <= self.stop_bandwidth)
            frequency, signal_fft = frequency[filter_frequency], matrix_dark_fft[i, :][filter_frequency]
            dark_norm = np.mean(np.abs(signal_fft) ** 2)
            # Then light
            filter_frequency = (frequency_light >= self.min_THz_frequency) & (frequency_light <= self.max_THz_frequency)
            frequency, signal_fft = frequency_light[filter_frequency], matrix_light_fft[i, :][filter_frequency]
            if i == 1:
                label_str = f"{i} average"
            else:
                label_str = f"{i} averages"
            ax[1].plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
                       color="tab:blue",
                       alpha=alpha_values[j],
                       label=label_str)
        filter_frequency = (frequency_light >= self.min_THz_frequency) & (frequency_light <= self.max_THz_frequency)
        frequency, signal_fft = frequency_light[filter_frequency], matrix_light_fft[-1, :][filter_frequency]
        ax[1].plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
                   color="tab:blue",
                   alpha=0.8,
                   label=f"{data['light']['number_of_traces']} THz averages")
        ax[0, 1].xaxis.set_major_formatter(EngFormatter(unit='Hz'))
        ax[0, 1].set_ylabel(r"Power spectrum [dB]")
        ax[0, 1].set_xlabel("Frequency")
        ax[0, 1].grid()
        if water_absorption_lines:
            h2o = np.loadtxt("WaterAbsorptionLines.csv", skiprows=8)
            filter_frequency = (h2o[:, 0] >= self.min_THz_frequency) & (h2o[:, 0] <= self.max_THz_frequency)
            # Filter for specified frequency range
            h2o = h2o[filter_frequency, :]
            [ax[1].axvline(_x, linewidth=1, color='#1f857c', alpha=0.3) for _x in h2o[:-1, 0]]
            ax[1].axvline(h2o[-1, 0], linewidth=1, color='#1f857c', alpha=0.3, label=r"$H_{2}O$ absorption lines")
        ax[1].legend()

        # Third plot, effect of averaging
        filter_dark = (frequency_dark >= self.start_bandwidth) & (frequency_dark <= self.stop_bandwidth)
        filter_light = (frequency_light >= self.start_bandwidth) & (frequency_light <= self.stop_bandwidth)
        dynamic_range = np.max(np.abs(matrix_light_fft[:, filter_light]) ** 2, axis=1) \
                        / np.mean(np.abs(matrix_dark_fft[:, filter_dark]) ** 2, axis=0)
        ax[2].scatter(np.arange(1, data["light"]["number_of_traces"] + 1),
                      10 * np.log10(dynamic_range),
                      color="tab:blue",
                      label=f"Max. {int(np.round(np.max(dynamic_range)))} dB")
        ax[2].set_xscale('log')
        ax[2].set_yscale('log')
        ax[2].set_xlabel(r"$N$, cumulative averaged traces")
        ax[2].set_ylabel("Peak Dynamic range [dB], frequency domain")
        ax[2].grid(True)

    def extract_bandwidth(self, data):
        # Find frequency range, where averaged light traces in frequency domain are
        # at least 2 x bigger than averaged dark traces
        frequency_dark, signal_fft_dark = calc_fft(data["dark"]["light_time"],
                                                   data["dark"]["average"]["time_domain"])
        frequency_light, signal_fft_light = calc_fft(data["light"]["light_time"],
                                                     data["light"]["average"]["time_domain"])
        filter_frequency_dark = (frequency_dark >= self.min_THz_frequency) & (frequency_dark <= self.max_THz_frequency)
        filter_frequency_light = (frequency_light >= self.min_THz_frequency) & (
                frequency_light <= self.max_THz_frequency)
        if np.all(np.diff(frequency_dark[filter_frequency_dark]) > 0):
            power_dark = np.interp(frequency_light[filter_frequency_light], filter_frequency_dark,
                                   np.abs(signal_fft_dark) ** 2)
            power_light = np.abs(signal_fft_light[filter_frequency_light]) ** 2
            frequency = frequency_light[filter_frequency_light]
        else:
            raise ValueError("The frequency axis is not strictly increasing, "
                             "which is a necessity for numpy's interpolation function.")
        peak_location = np.argmax(power_light)
        left_side = (frequency >= self.min_THz_frequency) & (frequency <= peak_location)
        right_side = (frequency > peak_location) & (frequency <= self.max_THz_frequency)
        start_idx = np.where(power_light[np.argsort(frequency[left_side])[::-1]] >
                             2 * power_dark[np.argsort(frequency[left_side])[::-1]])[0][0]
        stop_idx = np.where(power_light[np.argsort(frequency[right_side])] >
                            2 * power_dark[np.argsort(frequency[right_side])])[0][0]
        self.start_bandwidth = frequency[np.argsort(frequency[left_side])[::-1]][start_idx]
        self.stop_bandwidth = frequency[np.argsort(frequency[right_side])][stop_idx]
        self.bandwidth = self.stop_bandwidth - self.start_bandwidth
