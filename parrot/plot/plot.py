"""Module for plotting the processed THz data.

Besides displaying the data in time and frequency domain,
further metrics like signal-to-noise ratio (SNR) and dynamic range (DR) are calculated.
There is also the option if water vapor absorption lines should be displayed in the frequency domain.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from importlib import resources

# Own library
from ..process import post_process_data
from ..config import config


def _calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft


def _cumulated_mean_fft(data, x_samples):
    # Cumulative mean of all single traces, elegant but also computationally heavy
    # matrix = np.cumsum(data["single_traces"], axis=1) / np.arange(1, data["number_of_traces"] + 1)
    # ---
    # Instead, only do the mean of the supplied indices in x_samples
    matrix = np.zeros((data["single_traces"].shape[0], len(x_samples)))
    for i, trace_range in enumerate(x_samples):
        matrix[:, i] = np.mean(data["single_traces"][:, :trace_range], axis=1)
    matrix = np.fft.rfft(matrix, axis=0).T
    return matrix


def _plot_time_domain(data, ax, fill_between=False):
    if "dark1" in data.keys() and "dark2" in data.keys() and "dark" not in data.keys():
        raise NotImplementedError(
            "You supplied two dark measurements but did not execute `correct_systematic_errors()` "
            "of the `process`-module yet.")
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
        if fill_between:
            std_traces = np.std(data[mode]["single_traces"], axis=1)
            ax.fill_between(data[mode]["light_time"],
                            data[mode]["average"]["time_domain"] - std_traces,
                            data[mode]["average"]["time_domain"] + std_traces,
                            color=color,
                            alpha=0.3,
                            label=f"Standard deviation of {label_text} traces")
    ax.grid(True)
    # If the data got artificially extended with zeros in timedomain,
    # we want to limit the x-axis in timedomain and just zoom-in on real data.
    # The presented data is not cut off, you can still pan the axis window.
    data_start = data["light"]["average"]["time_domain"].nonzero()[0][0]
    data_stop = data["light"]["average"]["time_domain"].nonzero()[0][-1]
    ax.set_xlim([data["light"]["light_time"][data_start], data["light"]["light_time"][data_stop]])
    ax.legend(loc='upper right')
    ax.set_xlabel("Light time")
    ax.set_ylabel("Amplitude")
    ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax.yaxis.set_major_formatter(EngFormatter(unit='V'))
    return ax


def _plot_log_freq_domain(data, ax, min_THz_frequency, max_THz_frequency, stack_of_averages=False):
    matrix_dark_fft = None
    matrix_light_fft = None
    dark_fft = None
    # Prepare data, for an accumulated mean fo all single-traces calculate the FFT and dynamic range
    if not any(item.startswith("window") for item in data["applied_functions"]):
        config.logger.warn("It seems that you did not apply a window function to the data, "
                           "which will result in artifacts when using FFT."
                           "Please use `data = parrot.post_process_data.window(data)` before plotting.")
    if stack_of_averages:
        frequency_dark = data["dark"]["frequency"]
        dark_fft = data["dark"]["average"]["frequency_domain"]
        frequency_light = data["light"]["frequency"]
        light_fft = data["light"]["average"]["frequency_domain"]
    else:
        frequency_dark, dark_fft = _calc_fft(data["dark"]["light_time"], data["dark"]["average"]["time_domain"])
        frequency_light, light_fft = _calc_fft(data["light"]["light_time"], data["light"]["average"]["time_domain"])
    filter_frequency_dark = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
            frequency_dark <= data["statistics"]["bandwidth_stop"])
    dark_norm = np.mean(np.abs(dark_fft[filter_frequency_dark]) ** 2)

    # Starting to plot
    ax.axvline(data["statistics"]["bandwidth_start"], linestyle="--", color="black", alpha=0.5)
    if stack_of_averages:
        bandwidth_label = None
    else:
        bandwidth_label = (f"Bandwidth > {data['statistics']['bandwidth_threshold_dB']} dB: " +
                           f"{EngFormatter('Hz', places=1)(data['statistics']['bandwidth'])}")
    ax.axvline(data["statistics"]["bandwidth_stop"], linestyle="--", color="black", alpha=0.5, label=bandwidth_label)
    filter_frequency = (frequency_light >= min_THz_frequency) & (frequency_light <= max_THz_frequency)
    frequency, signal_fft = frequency_light[filter_frequency], light_fft[filter_frequency]
    ax.plot(frequency, 10 * np.log10(np.abs(signal_fft) ** 2 / dark_norm),
            color="tab:orange",
            alpha=0.8,
            label=f"{data['light']['number_of_traces']} averaged THz traces")
    if stack_of_averages:
        # To not clutter the plot with too many curves, just plot an accumulated average of multiples of curves,
        # means:
        # If the dataframe contains 573 single THz traces,
        # then plot all traces, 10% of all traces, 1 % of all traces, and so on.
        trace_range = [data["light"]['number_of_traces']]
        while trace_range[-1] > 9:
            trace_range.append(int(np.round(trace_range[-1] / 10)))
        # Delete first number (full amount of traces), since it is already plotted
        trace_range = trace_range[1:]
        trace_range = trace_range[::-1]
        # Thanks to Malte, instead of a variation of alpha-values,
        # the plasma colormap is used to differentiate the curves from each other
        # It is restricted to a maximum of 0.75, to not reach to very bright/light yellow colors,
        # which are difficult to see on white background.
        colors = plt.cm.plasma(np.linspace(0, 0.5, len(trace_range)))
        colors = colors[::-1]
        for j, i in enumerate(trace_range[::-1]):
            # Select a temporary subset of the dataframe, first dark
            filter_frequency = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
                    frequency_dark <= data["statistics"]["bandwidth_stop"])
            frequency, signal_fft = _calc_fft(data["dark"]["light_time"],
                                              np.mean(data["dark"]["single_traces"][:, :i - 1], axis=1))
            dark_norm = np.mean(np.abs(signal_fft) ** 2)
            # Then light
            filter_frequency = (frequency_light >= min_THz_frequency) & (frequency_light <= max_THz_frequency)
            frequency, signal_fft = _calc_fft(data["light"]["light_time"],
                                              np.mean(data["light"]["single_traces"][:, :i - 1], axis=1))
            if i == 1:
                label_str = f"{i} THz trace"
            else:
                label_str = f"{i} averaged THz traces"
            ax.plot(frequency[filter_frequency], 10 * np.log10(np.abs(signal_fft[filter_frequency]) ** 2 / dark_norm),
                    color=colors[j],
                    alpha=0.8,
                    label=label_str)
    if "dark" in data.keys():
        dark_norm = np.mean(np.abs(dark_fft[filter_frequency]) ** 2)
        filter_frequency = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)
        ax.plot(frequency_dark[filter_frequency],
                10 * np.log10(np.abs(dark_fft[filter_frequency]) ** 2 / dark_norm),
                zorder=1.9,
                color="black",
                alpha=0.8,
                label=f"{data['dark']['number_of_traces']} dark traces averaged")
    ax.xaxis.set_major_formatter(EngFormatter(unit='Hz'))
    ax.set_xlabel("Frequency")
    ax.set_ylabel(r"Power spectrum (dB)")
    ax.legend(loc="upper right")
    ax.grid(True)
    return ax


def _plot_water_absorption_lines(data, ax):
    with resources.path("parrot.plot.h2o", "WaterAbsorptionLines.csv") as file:
        h2o = np.loadtxt(file, delimiter=",", skiprows=8)
    filter_frequency = (h2o[:, 0] >= data["statistics"]["bandwidth_start"]) & (
                h2o[:, 0] <= data["statistics"]["bandwidth_stop"])
    # Filter for specified frequency range
    h2o = h2o[filter_frequency, :]
    alpha_values = np.linspace(0.4, 0.05, len(h2o) - 1)
    h2o_sorted_freq = h2o[np.argsort(h2o[:, 2]), 0]
    ax.axvline(h2o_sorted_freq[0], linewidth=1, color='tab:blue', alpha=0.5, label=r"$H_{2}O$ absorption lines")
    [ax.axvline(_x, linewidth=1, color='tab:blue', alpha=alpha_values[i]) for i, _x in enumerate(h2o_sorted_freq[1:])]
    return ax


def extended_multi_cycle(data,
                         min_THz_frequency=0e12,
                         max_THz_frequency=10e12,
                         threshold_dB=10,
                         snr_timedomain=False,
                         water_absorption_lines=True,
                         figsize=None,
                         vertical_stacked_plots=True):
    if figsize is None:
        # This is a good size for a 16:9 (PowerPoint) presentation.
        # figsize only accepts inches, so values in [cm] are converted.
        # Tip: Set vertical_stacked_plots=False to have the plots next to each other
        figsize = (32.5/2.54, 15.5/2.54)
    if vertical_stacked_plots:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    # First subplot, time domain
    ax = axs[0]
    ax = _plot_time_domain(data, ax, fill_between=True)
    if snr_timedomain:
        # Filter all zeros in array (from windowing, extra padding, etc.) since we cannot divide by 0
        std_traces = np.std(data["light"]["single_traces"], axis=1)
        filter_zeros = (std_traces == 0)
        # Since the filter is True, when there is zero, we need to use the opposite of that.
        snr_timedomain_max = np.nanmax(np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) /
                                       std_traces[~filter_zeros])
        snr_timedomain = ax.twinx()
        snr_timedomain.plot(data["light"]["light_time"],
                            np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) / std_traces[
                                ~filter_zeros],
                            color="tab:green",
                            alpha=0.4)
        snr_timedomain.set_ylabel("SNR")
        ax.scatter([], [], c="tab:green", label=f'SNR, timedomain, max: {int(snr_timedomain_max)}')

    # Second subplot, frequency-domain
    data = post_process_data.get_statistics(data,
                                            min_THz_frequency=min_THz_frequency,
                                            max_THz_frequency=max_THz_frequency,
                                            threshold_dB=threshold_dB)

    ax = axs[1]
    ax = _plot_log_freq_domain(data, ax, min_THz_frequency, max_THz_frequency, stack_of_averages=True)
    if water_absorption_lines:
        ax = _plot_water_absorption_lines(data, ax)

    # Third plot, effect of averaging
    ax = axs[2]
    # TODO: Show in documentation why we expect a linear relationship for effective averaging
    frequency_dark = data["dark"]["frequency"]
    frequency_light = data["light"]["frequency"]

    filter_dark = (frequency_dark >= data["statistics"]["bandwidth_start"]) & (
            frequency_dark <= data["statistics"]["bandwidth_stop"])
    filter_light = (frequency_light >= data["statistics"]["bandwidth_start"]) & (
            frequency_light <= data["statistics"]["bandwidth_stop"])
    min_of_max_traces = np.min([data["light"]["number_of_traces"], data["dark"]["number_of_traces"]])
    # Calculating the dynamic range for every (cumulated) trace can be very computationally heavy
    # Instead, if the min_of_max_traces is e.g., 3720, calculate an array with 10^n
    # The following formula would for example produce x_samples = [1,10,100], since we use later log-axis
    x_samples = np.logspace(0, int(np.log10(min_of_max_traces)) - 1, int(np.log10(min_of_max_traces)))
    # Repeat every item 10x and take the cumulative sum. Attach at the end the total number of traces
    x_samples = np.concatenate((np.cumsum(np.repeat(x_samples, 10)), np.array([min_of_max_traces])))
    # We can only use integer amounts for averaged traces, thus the forced type to int.
    # This gives many double entries due to rounding, thus the use of np.unique().
    x_samples = np.unique(np.logspace(0, np.log10(min_of_max_traces), 100, dtype=int))
    matrix_light_dr = _cumulated_mean_fft(data["light"], x_samples)
    matrix_dark_dr = _cumulated_mean_fft(data["dark"], x_samples)
    dynamic_range = np.max(np.abs(matrix_light_dr[:, filter_light]) ** 2, axis=1) \
                    / np.mean(np.abs(matrix_dark_dr[:, filter_dark]) ** 2, axis=1)
    range_for_color = np.logspace(np.log10(len(dynamic_range)), np.log10(1), len(dynamic_range))
    range_for_color -= 1
    range_for_color = 0.75 * range_for_color / np.max(range_for_color) + 0.25

    ax.scatter(x_samples,
               dynamic_range,
               color=plt.cm.plasma_r(range_for_color))
    ax.scatter([], [], color="tab:orange", label=f"Max. {int(np.round(10 * np.log10(np.max(dynamic_range))))} dB")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"$N$, cumulative averaged traces")
    ax.set_ylabel("Peak dynamic range\nfrequency domain")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    fig.align_ylabels(axs)
    plt.tight_layout()
    plt.show(block=False)


def simple_multi_cycle(data,
                       min_THz_frequency=0e12,
                       max_THz_frequency=10e12,
                       threshold_dB=10,
                       figsize=None,
                       vertical_stacked_plots=True,
                       water_absorption_lines=True,
                       debug=False):
    if debug:
        config.set_debug(True)
    else:
        config.set_debug(False)
    if figsize is None:
        figsize = (12, 8)
    if vertical_stacked_plots:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    else:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    fig_title = ""
    # First subplot, time domain
    ax = axs[0]
    ax = _plot_time_domain(data, ax, fill_between=False)

    # Calculating SNR in time domain
    # Filter all zeros in array (from windowing, extra padding, etc.) since we cannot divide by 0
    std_traces = np.std(data["light"]["single_traces"], axis=1)
    filter_zeros = (std_traces == 0)
    # Since the filter is True, when there is zero, we need to use the opposite of that.
    snr_timedomain_max = np.nanmax(np.abs(data["light"]["average"]["time_domain"][~filter_zeros]) /
                                   std_traces[~filter_zeros])
    fig_title += r"$\mathrm{SNR}_{\mathrm{max}}$" + f" (timedomain): {int(snr_timedomain_max)}"

    ax = axs[1]
    # Second subplot, frequency-domain
    data = post_process_data.get_statistics(data,
                                            min_THz_frequency=min_THz_frequency,
                                            max_THz_frequency=max_THz_frequency,
                                            threshold_dB=threshold_dB)
    frequency_dark, dark_fft = _calc_fft(data["dark"]["light_time"], data["dark"]["average"]["time_domain"])
    frequency_light, light_fft = _calc_fft(data["light"]["light_time"], data["light"]["average"]["time_domain"])
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
    ax = _plot_log_freq_domain(data, ax, min_THz_frequency, max_THz_frequency, stack_of_averages=False)
    if water_absorption_lines:
        ax = _plot_water_absorption_lines(data, ax)

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
