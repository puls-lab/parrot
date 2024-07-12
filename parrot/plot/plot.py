import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from pathlib import Path
from scipy.optimize import brentq
from scipy.stats import norm
from IPython.display import display, clear_output

# Own library
from ..process import post_process_data
from ..config import config


def calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft


def cumulated_mean_fft(data):
    dt = (data["light_time"][-1] - data["light_time"][0]) / (len(data["light_time"]) - 1)
    frequency = np.fft.rfftfreq(len(data["light_time"]), dt)
    # Cumulative mean of all single traces
    matrix = np.cumsum(data["single_traces"], axis=1) / np.arange(1, data["number_of_traces"] + 1)
    matrix = np.fft.rfft(matrix, axis=0).T
    return frequency, matrix


def extended_multi_cycle(data,
                         min_THz_frequency=0e12,
                         max_THz_frequency=10e12,
                         threshold_dB=10,
                         figsize=None,
                         snr_timedomain=False,
                         water_absorption_lines=True):
    if figsize is None:
        figsize = (12, 8)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    # First subplot, time domain
    for mode in data.keys():
        if mode == "light":
            label_text = "THz"
            color = "tab:orange"
        elif mode == "dark":
            label_text = "dark"
            color = "black"
        else:
            continue
        std_traces = np.std(data[mode]["single_traces"], axis=1)
        ax[0].plot(data[mode]["light_time"],
                   data[mode]["average"]["time_domain"],
                   color=color,
                   alpha=0.8,
                   label=f"Average of {data[mode]['number_of_traces']} {label_text} traces")
        ax[0].fill_between(data[mode]["light_time"],
                           data[mode]["average"]["time_domain"] - std_traces,
                           data[mode]["average"]["time_domain"] + std_traces,
                           color=color,
                           alpha=0.3,
                           label=f"Standard deviation of {label_text} traces")
    ax[0].grid(True)
    # If the data got artificially extended with zeros in timedomain,
    # we want to limit the x-axis in timedomain and just zoom-in on real data.
    # The presented data is not cut off, you can still pan the axis window.
    data_start = data["light"]["average"]["time_domain"].nonzero()[0][0]
    data_stop = data["light"]["average"]["time_domain"].nonzero()[0][-1]
    ax[0].set_xlim([data["light"]["light_time"][data_start], data["light"]["light_time"][data_stop]])
    if snr_timedomain:
        # Filter all zeros in array (from windowing, extra padding, etc.) since we cannot divide by 0
        std_traces = np.std(data["light"]["single_traces"], axis=1)
        filter_zeros = (std_traces == 0)
        # Since the filter is True, when there is zero, we need to use the opposite of that.
        snr_timedomain_max = np.nanmax(np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) /
                                       std_traces[~filter_zeros])
        snr_timedomain = ax[0].twinx()
        snr_timedomain.plot(data["light"]["light_time"],
                            np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) / std_traces[
                                ~filter_zeros],
                            color="tab:green",
                            alpha=0.4)
        snr_timedomain.set_ylabel("SNR")
        ax[0].scatter([], [], c="tab:green", label=f'SNR, timedomain, max: {int(snr_timedomain_max)}')
    ax[0].legend(loc='upper right')
    ax[0].xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax[0].set_xlabel("Light time")
    ax[0].set_ylabel("Amplitude")
    ax[0].yaxis.set_major_formatter(EngFormatter(unit='V'))

    # Second subplot, frequency-domain
    data = post_process_data.get_statistics(data, min_THz_frequency, max_THz_frequency, threshold_dB)
    # Prepare data, for an accumulated mean fo all single-traces calculate the FFT and dynamic range
    if not any(item.startswith("window") for item in data["applied_functions"]):
        config.logger.warn("It seems that you did not apply a window function to the data, "
                           "which will result in artifacts when using FFT."
                           "Please use `data = parrot.post_process_data.window(data)` before plotting.")
    frequency_dark, matrix_dark_fft = cumulated_mean_fft(data["dark"])
    frequency_light, matrix_light_fft = cumulated_mean_fft(data["light"])
    filter_frequency = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
            frequency_dark <= data["statistics"]["bandwidth_stop"])
    dark_norm = np.mean(np.abs(matrix_dark_fft[-1, :][filter_frequency]) ** 2)

    ax[1].axvline(data["statistics"]["bandwidth_start"], linestyle="--", color="black", alpha=0.5)
    ax[1].axvline(data["statistics"]["bandwidth_stop"], linestyle="--", color="black", alpha=0.5)
    filter_frequency = (frequency_light >= min_THz_frequency) & (frequency_light <= max_THz_frequency)
    frequency, signal_fft = frequency_light[filter_frequency], matrix_light_fft[-1, :][filter_frequency]
    ax[1].plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
               color="tab:orange",
               alpha=0.8,
               label=f"{data['light']['number_of_traces']} averaged THz traces")
    # To not clutter the plot with too many curves, just plot an accumulated average of multiples of curves,
    # means:
    # If the dataframe contains 573 single THz traces,
    # then plot the average (and FFT) of 1, 10, 100 and all 573 traces.
    trace_range = np.logspace(0,
                              np.floor(np.log10(data["light"]['number_of_traces'])),
                              num=1 + int(np.floor(np.log10(data["light"]["number_of_traces"])))
                              ).astype(int)

    # Make the curves which contain less averaged spectra more transparent
    alpha_values = np.arange(0.8 - 0.15 * len(trace_range), 1, 0.15)[::-1]
    for j, i in enumerate(trace_range[::-1]):
        # Select a temporary subset of the dataframe, first dark
        filter_frequency = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
                frequency_dark <= data["statistics"]["bandwidth_stop"])
        frequency, signal_fft = frequency_dark[filter_frequency], matrix_dark_fft[i - 1, :][filter_frequency]
        dark_norm = np.mean(np.abs(signal_fft) ** 2)
        # Then light
        filter_frequency = (frequency_light >= min_THz_frequency) & (frequency_light <= max_THz_frequency)
        frequency, signal_fft = frequency_light[filter_frequency], matrix_light_fft[i - 1, :][filter_frequency]
        if i == 1:
            label_str = f"{i} THz trace"
        else:
            label_str = f"{i} averaged THz traces"
        ax[1].plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
                   color="tab:orange",
                   alpha=alpha_values[j],
                   label=label_str)
    dark_norm = np.mean(np.abs(matrix_dark_fft[-1, :][filter_frequency]) ** 2)
    filter_frequency = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)
    ax[1].plot(frequency_dark[filter_frequency],
               10 * np.log10(np.abs(matrix_dark_fft[-1, :][filter_frequency]) ** 2 / dark_norm),
               zorder=1.9,
               color="black",
               alpha=0.8,
               label=f"{data['dark']['number_of_traces']} dark traces averaged")
    if water_absorption_lines:
        script_path = Path(__file__).resolve().parent
        h2o = np.loadtxt(script_path / "WaterAbsorptionLines.csv", delimiter=",", skiprows=8)
        filter_frequency = (h2o[:, 0] >= data["statistics"]["bandwidth_start"]) & (
                h2o[:, 0] <= data["statistics"]["bandwidth_stop"])
        # Filter for specified frequency range
        h2o = h2o[filter_frequency, :]
        alpha_values = np.linspace(0.4, 0.05, len(h2o) - 1)
        h2o_sorted_freq = h2o[np.argsort(h2o[:, 2]), 0]
        ax[1].axvline(h2o_sorted_freq[0], linewidth=1, color='tab:blue', alpha=0.5, label=r"$H_{2}O$ absorption "
                                                                                          "lines")
        [ax[1].axvline(_x, linewidth=1, color='tab:blue', alpha=alpha_values[i]) for i, _x in
         enumerate(h2o_sorted_freq[1:])]
    ax[1].xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax[1].set_ylabel(r"Power spectrum (dB)")
    ax[1].set_xlabel("Frequency")
    ax[1].grid(True)
    ax[1].legend(loc="upper right")

    # Third plot, effect of averaging
    # TODO: Show in documentation why we expect a linear relationship for effective averaging
    filter_dark = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
            frequency_dark <= data["statistics"]["bandwidth_stop"])
    filter_light = (frequency_light >= data["statistics"]["bandwidth_start"]) & (
            frequency_light <= data["statistics"]["bandwidth_stop"])
    min_of_max_traces = np.min([data["light"]["number_of_traces"], data["dark"]["number_of_traces"]])
    dynamic_range = np.max(np.abs(matrix_light_fft[:min_of_max_traces, filter_light]) ** 2, axis=1) \
                    / np.mean(np.abs(matrix_dark_fft[:min_of_max_traces, filter_dark]) ** 2, axis=1)
    ax[2].scatter(np.arange(1, min_of_max_traces + 1),
                  dynamic_range,
                  color="tab:blue",
                  label=f"Max. {int(np.round(10 * np.log10(np.max(dynamic_range))))} dB")
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r"$N$, cumulative averaged traces")
    ax[2].set_ylabel("Peak dynamic range\nfrequency domain")
    ax[2].legend(loc="upper left")
    ax[2].grid(True, which="both")
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show(block=False)


def simple_multi_cycle(data,
                       min_THz_frequency=0e12,
                       max_THz_frequency=10e12,
                       threshold_dB=10,
                       figsize=None,
                       water_absorption_lines=True,
                       debug=False):
    if debug:
        config.set_debug(True)
    else:
        config.set_debug(False)
    if figsize is None:
        figsize = (12, 8)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    fig_title = ""
    ax = axs[0]
    if "dark1" in data.keys() and "dark2" in data.keys() and "dark" not in data.keys():
        raise NotImplementedError(
            "You supplied two dark measurements but did not execute `correct_systematic_errors()` "
            "of the `process`-module yet.")
    # First subplot, time domain
    for mode in data.keys():
        if mode == "light":
            label_text = "THz"
            color = "tab:orange"
        elif mode == "dark":
            label_text = "dark"
            color = "black"
        else:
            continue
        ax.plot(data[mode]["light_time"],
                data[mode]["average"]["time_domain"],
                color=color,
                alpha=0.8,
                label=f"Average of {data[mode]['number_of_traces']} {label_text} traces")
    ax.grid(True)
    # If the data got artificially extended with zeros in time domain,
    # we want to limit the x-axis in time domain to just zoom on the data.
    # The data is not cut off, you can still pan the axis window,
    # but it is outside the selected range just zero.
    data_start = data["light"]["average"]["time_domain"].nonzero()[0][0]
    data_stop = data["light"]["average"]["time_domain"].nonzero()[0][-1]
    ax.set_xlim([data["light"]["light_time"][data_start], data["light"]["light_time"][data_stop]])

    # Calculating SNR in time domain
    # Filter all zeros in array (from windowing, extra padding, etc.) since we cannot divide by 0
    std_traces = np.std(data["light"]["single_traces"], axis=1)
    filter_zeros = (std_traces == 0)
    # Since the filter is True, when there is zero, we need to use the opposite of that.
    snr_timedomain_max = np.nanmax(np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) /
                                   std_traces[~filter_zeros])

    fig_title += r"$\mathrm{SNR}_{\mathrm{max}}$" + f" (timedomain): {int(snr_timedomain_max)}"
    ax.legend(loc='upper right')
    ax.set_xlabel("Light time")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax.yaxis.set_major_formatter(EngFormatter(unit='V'))

    ax = axs[1]
    # Second subplot, frequency-domain
    data = post_process_data.get_statistics(data,
                                            min_THz_frequency=min_THz_frequency,
                                            max_THz_frequency=max_THz_frequency,
                                            threshold_dB=threshold_dB)
    # Prepare data, for an accumulated mean fo all single-traces calculate the FFT and dynamic range
    if not any(item.startswith("window") for item in data["applied_functions"]):
        config.logger.warn("It seems that you did not apply a window function to the data, "
                           "which will result in artifacts when using FFT."
                           "Please use `data = parrot.post_process_data.window(data)` before plotting.")
    frequency_dark, dark_fft = calc_fft(data["dark"]["light_time"], data["dark"]["average"]["time_domain"])
    frequency_light, light_fft = calc_fft(data["light"]["light_time"], data["light"]["average"]["time_domain"])
    filter_frequency_dark = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
            frequency_dark <= data["statistics"]["bandwidth_stop"])
    dark_norm = np.mean(np.abs(dark_fft[filter_frequency_dark]) ** 2)
    filter_frequency = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)
    frequency, signal_fft = frequency_light[filter_frequency], light_fft[filter_frequency]
    amplitude_normed = np.abs(signal_fft) - np.mean(np.abs(dark_fft[filter_frequency_dark]))
    amplitude_normed /= np.max(amplitude_normed)
    ax.plot(frequency, amplitude_normed,
            color="tab:orange",
            alpha=0.8,
            label=f"{data['light']['number_of_traces']} averaged THz traces")
    ax.annotate(text='',
                xy=(data["statistics"]["fwhm_start"], 0.5),
                xytext=(data["statistics"]["fwhm_stop"], 0.5),
                arrowprops=dict(arrowstyle='<->', shrinkA=0, shrinkB=0))
    # center_position = (data["statistics"]["fwhm_stop"] - data["statistics"]["fwhm_start"]) / 2
    # center_position += data["statistics"]["fwhm_start"]
    # ax.text(center_position, 0.6, f'FWHM\n{EngFormatter("Hz",places=1)(data["statistics"]["fwhm"])}')
    # TODO: Make double arrow symbol in legend more visible
    ax.scatter([], [], color="black", marker=r"$\longleftrightarrow$", s=120,
               label=f'FWHM = {EngFormatter("Hz", places=1)(data["statistics"]["fwhm"])}')
    ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax.set_ylabel("Amplitude (linear & norm.)")
    ax.set_xlabel("Frequency")
    ax.grid(True)
    ax.legend()

    ax = axs[2]
    if "dark" in data.keys():
        mode = "dark"
        ax.plot(frequency_dark[filter_frequency],
                10 * np.log10(np.abs(dark_fft[filter_frequency]) ** 2 / dark_norm),
                color="black",
                alpha=0.8,
                label=f"{data[mode]['number_of_traces']} dark traces averaged")
    ax.axvline(data["statistics"]["bandwidth_start"],
               linestyle="--",
               color="black",
               alpha=0.5)
    ax.axvline(data["statistics"]["bandwidth_stop"],
               linestyle="--",
               color="black",
               alpha=0.5,
               label=f"Bandwidth > {data['statistics']['bandwidth_threshold_dB']} dB: "
                     f"{EngFormatter('Hz', places=1)(data['statistics']['bandwidth'])}")
    ax.plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
            color="tab:orange",
            alpha=0.8,
            label=f"{data['light']['number_of_traces']} averaged THz traces")
    ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax.set_ylabel(r"Power spectrum (dB)")
    ax.set_xlabel("Frequency")
    ax.grid(True)
    if water_absorption_lines:
        script_path = Path(__file__).resolve().parent
        h2o = np.loadtxt(script_path / "WaterAbsorptionLines.csv", delimiter=",", skiprows=8)
        filter_frequency = (h2o[:, 0] >= data["statistics"]["bandwidth_start"]) & (
                h2o[:, 0] <= data["statistics"]["bandwidth_stop"])
        # Filter for specified frequency range
        h2o = h2o[filter_frequency, :]
        alpha_values = np.linspace(0.4, 0.05, len(h2o) - 1)
        h2o_sorted_freq = h2o[np.argsort(h2o[:, 2]), 0]
        ax.axvline(h2o_sorted_freq[0], linewidth=1, color='tab:blue', alpha=0.5, label=r"$H_{2}O$ absorption "
                                                                                       "lines")
        [ax.axvline(_x, linewidth=1, color='tab:blue', alpha=alpha_values[i]) for i, _x in
         enumerate(h2o_sorted_freq[1:])]
    ax.legend(loc="upper right")

    filter_dark = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
            frequency_dark <= data["statistics"]["bandwidth_stop"])
    filter_light = (frequency_light >= data["statistics"]["bandwidth_start"]) & (
            frequency_light <= data["statistics"]["bandwidth_stop"])
    dynamic_range = np.max(np.abs(light_fft[filter_light]) ** 2) / np.mean(np.abs(dark_fft[filter_dark]) ** 2)
    fig_title += r", $\mathrm{DR}_{\mathrm{peak}}$" + f" (freq.-domain): {int(np.round(10 * np.log10(dynamic_range)))} dB"
    fig.subplots_adjust(top=0.9)
    fig.suptitle(fig_title)
    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show(block=False)


def debug_lab_time_raw(data):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(data["time"], data["position"], label="Position, raw")
    ax[1].plot(data["time"], data["signal"], label="Signal, raw")
    return fig, ax


def debug_lab_time_filtered(data, lowcut_position, highcut_position, lowcut_signal, highcut_signal, fig, ax):
    my_label = "Position, filtered\n"
    if lowcut_position is not None:
        my_label += f"High-pass: {EngFormatter('Hz')(lowcut_position)}\n"
    if highcut_position is not None:
        my_label += f"Low-pass: {EngFormatter('Hz')(highcut_position)}\n"
    my_label = my_label[:-1]
    if lowcut_position is None and highcut_position is None:
        my_label = "Position, untouched"
    ax[0].plot(data["time"], data["position"], label=my_label)

    my_label = "Signal, filtered\n"
    if lowcut_signal is not None:
        my_label += f"High-pass: {EngFormatter('Hz')(lowcut_signal)}\n"
    if highcut_signal is not None:
        my_label += f"Low-pass: {EngFormatter('Hz')(highcut_signal)}\n"
    my_label = my_label[:-1]
    if lowcut_signal is None and highcut_signal is None:
        my_label = "Signal, untouched"
    ax[1].plot(data["time"], data["signal"], label=my_label)

    ax[0].xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax[0].yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax[1].yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[0].grid(True)
    ax[1].grid(True)
    ax[0].set_title("Filtering of position and signal")
    ax[1].set_xlabel("Lab time")
    if data["number_of_traces"] > 10:
        ax[0].set_xlim([0, data["time"][data["trace_cut_index"][10]]])
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


def debug_position_cut(data, dataset_name):
    fig, ax = plt.subplots()
    ax.plot(data["position"], label="Position")
    [ax.axvline(x, color="black", alpha=0.8) for x in data["trace_cut_index"]]
    ax.axvline(np.nan, color="black", alpha=0.8, label="Cut index")
    ax.legend(loc="upper right")
    if data["number_of_traces"] > 10:
        # If more than 10 single traces are detected, restrict x-axis to only show first ten,
        # or you get a wall of vertical lines indicating the extrema of the positional data
        ax.set_xlim([0, data["trace_cut_index"][10]])
    if dataset_name is None:
        ax.set_title(f"Cutting dataset into single traces")
    else:
        # If name of dataset like "light", "dark1", etc. are given, place it into the axis title.
        ax.set_title(f"Cutting {dataset_name} dataset into single traces")
    ax.set_xlabel("Time samples")
    ax.set_ylabel("Position")
    ax.yaxis.set_major_formatter(EngFormatter("V"))
    ax.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


def debug_optimizing_delay(fig, axs=None, iteration_step=None, delay=None, error=None):
    if fig is None:
        fig, axs = plt.subplots(nrows=2, ncols=1)
        for ax in axs:
            ax.set_xlabel("Iteration")
            ax.grid(True)
        axs[1].set_xlabel("Iteration")
        axs[0].set_ylabel("Delay (timesteps)")
        axs[1].set_ylabel("Error (a.u.)")
        axs[0].set_title("Optimizing delay between position- and THz-signal")
        return fig, axs

    ax = axs[0]
    _ = ax.plot(iteration_step, delay, ".-", color="tab:blue")

    ax = axs[1]
    _ = ax.plot(iteration_step, error, ".-", color="tab:orange")
    _ = display(fig)

    _ = clear_output(wait=True)
    _ = plt.tight_layout()
    _ = plt.pause(0.05)
    _ = plt.show(block=False)
    return fig, axs


def debug_no_delay_compensation(data, original_time, position_interpolated, interpolated_delay, consider_all_traces):
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    split_pos = np.split(data["position"], data["trace_cut_index"])
    split_sig = np.split(data["signal"], data["trace_cut_index"])
    for i in range(1, 10):
        if i % 2:
            ax[0].plot(data["scale"] * split_pos[i],
                       split_sig[i],
                       color="tab:blue", alpha=0.8)
        else:
            ax[0].plot(data["scale"] * split_pos[i],
                       split_sig[i],
                       color="tab:orange", alpha=0.8)
    ax[0].plot([], [], color="tab:blue", alpha=0.8, label="Signal forward")
    ax[0].plot([], [], color="tab:orange", alpha=0.8, label="Signal backward")
    ax[0].xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax[0].yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax[0].set_title('Delay = 0 Sa')
    ax[0].legend(loc="upper right")

    # Calculate STD when no delay correction is applied
    new_position = position_interpolated(original_time)
    new_position = (new_position - np.nanmin(new_position)) / (np.nanmax(new_position) - np.nanmin(new_position))
    signal_matrix = np.zeros((data["interpolation_resolution"], data["number_of_traces"]))
    signal_matrix[:] = np.NaN

    i = 0
    for position, signal in zip(np.split(new_position, data["trace_cut_index"]), split_sig):
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

    ax[2].semilogy(data["light_time"], np.nanstd(signal_matrix, axis=1), color="black", label="Delay = 0 Sa")
    return fig, ax


def debug_with_delay_compensation(data, interpolated_delay, consider_all_traces, dataset_name,
                                  fig, ax):
    split_pos = np.split(data["position"], data["trace_cut_index"])
    split_sig = np.split(data["signal"], data["trace_cut_index"])
    for i in range(1, 10):
        if i % 2:
            ax[1].plot(data["scale"] * split_pos[i],
                       split_sig[i],
                       color="tab:blue", alpha=0.8)
        else:
            ax[1].plot(data["scale"] * split_pos[i],
                       split_sig[i],
                       color="tab:orange", alpha=0.8)
    ax[1].xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax[1].yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax[1].plot([], [], color="tab:blue", alpha=0.8, label="Signal forward")
    ax[1].plot([], [], color="tab:orange", alpha=0.8, label="Signal backward")
    delay_amount = data["delay_value"] * data["dt"]
    ax[1].set_title(
        f'Delay = {data["delay_value"]:.3f} Sa ' +
        f'({EngFormatter("s", places=1)(delay_amount)}) time delay compensation')
    ax[1].legend(loc="upper right")

    # Calculate STD when delay correction is applied
    new_position = data["position"]
    new_position = (new_position - np.nanmin(new_position)) / (np.nanmax(new_position) - np.nanmin(new_position))
    signal_matrix = np.zeros((data["interpolation_resolution"], data["number_of_traces"]))
    signal_matrix[:] = np.NaN
    i = 0
    for position, signal in zip(np.split(new_position, data["trace_cut_index"]), split_sig):
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
    ax[2].semilogy(data["light_time"], np.nanstd(signal_matrix, axis=1), color="tab:green",
                   label=f"Delay = {data['delay_value']:.3f} Sa")
    ax[2].legend(loc="upper right")
    ax[2].grid(True, which="both")
    ax[2].yaxis.set_major_formatter(EngFormatter("V"))
    if dataset_name is None:
        ax[2].set_title(f"Standard deviation")
    else:
        ax[2].set_title(f"Standard deviation of dataset {dataset_name}")
    ax[2].set_xlabel("Light time")
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


def debug_analysis_amplitude_jitter(data):
    # data is already the subset of the light-dataset
    fig, axs = plt.subplots(nrows=2, ncols=2)
    ax = axs[0, 0]
    # Calculate the amplitude peak of each single trace
    amplitude_peaks = np.max(data["single_traces"], axis=0)
    ax.plot(np.arange(1, data["number_of_traces"] + 1), amplitude_peaks, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Trace ID")
    ax.set_ylabel("Peak amplitude")
    ax.yaxis.set_major_formatter(EngFormatter("V"))
    ax.grid(True)

    ax = axs[1, 0]
    # Calculate jitter/zero-crossing between max. and min. for each single trace
    idx_min = np.min([np.argmin(data["average"]["time_domain"]), np.argmax(data["average"]["time_domain"])])
    idx_max = np.max([np.argmin(data["average"]["time_domain"]), np.argmax(data["average"]["time_domain"])])
    # Use (int) index as the basis of the x-axis for creating a CubicSpline instead of light time as x-axis,
    # to not run into trouble with interpolation of small numbers on the order of 1e-12, which is used in light time.
    x = np.arange(idx_min, idx_max)
    if not np.all(np.diff(x) > 0):
        raise ValueError("The index array needs to be monotonically increasing, otherwise np.interp will not work.")
    zero_crossing = np.zeros(data["single_traces"].shape[1])
    zero_crossing[:] = np.nan
    for i in range(data["single_traces"].shape[1]):
        y = data["single_traces"][idx_min:idx_max, i]
        # Use linear interpolation and not cubic_spline,
        # which can have slight deviations between x, shifting the zero-crossing
        f = lambda xnew: np.interp(xnew, x, y)
        root = brentq(f, idx_min, idx_max)
        # Convert (float) index to light_time value by linear interpolation
        zero_crossing[i] = np.interp(root, np.arange(len(data["light_time"])), data["light_time"])
    # Calculate zero-crossing/jitter with respect to the first trace
    zero_crossing -= zero_crossing[0]
    # Plot zero-corssing vs. trace ID
    ax.plot(np.arange(1, data["number_of_traces"] + 1), zero_crossing, color="tab:orange", alpha=0.8)
    ax.set_xlabel("Trace ID")
    ax.set_ylabel("Zero-crossing / jitter\nwith respect to the 1st trace")
    ax.yaxis.set_major_formatter(EngFormatter("s"))
    ax.grid(True)

    ax = axs[0, 1]
    # Plot amplitude as a histogram
    (mu, sigma) = norm.fit(amplitude_peaks)
    places = int(np.round(-np.log10(sigma / mu)))

    ax.hist(np.max(data["single_traces"], axis=0), density=False, rwidth=0.9, color="tab:blue", bins="auto")
    ax.xaxis.set_major_formatter(EngFormatter("V"))
    ax.set_xlabel("Peak amplitude")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{EngFormatter('V', places=places)(mu)}" +
                 r"$\pm$" +
                 f"{EngFormatter('V', places=places)(sigma)}" +
                 f"({sigma / mu:.1%})")

    ax = axs[1, 1]
    # Plot zero-crossing as a histogram
    (mu, sigma) = norm.fit(zero_crossing)
    ax.hist(zero_crossing, density=False, rwidth=0.9, color="tab:orange", bins="auto")
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.set_xlabel("Zero-crossing / jitter")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{EngFormatter('s', places=1)(mu)}" +
                 r"$\pm$" +
                 f"{EngFormatter('s', places=1)(sigma)}")

    plt.tight_layout()
    plt.show(block=False)
    return fig, ax
