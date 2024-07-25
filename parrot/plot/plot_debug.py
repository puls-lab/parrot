"""This module contains various plot-functions which are called when debug is set to True during processing."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.optimize import brentq
from scipy.stats import norm
# Own library
from ..process import post_process_data
from ..config import config


def lab_time_raw(data, figsize=None):
    if figsize is None:
        figsize = (8, 8)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)
    ax[0].plot(data["time"], data["position"], label="Position, raw")
    ax[1].plot(data["time"], data["signal"], label="Signal, raw")
    return fig, ax


def lab_time_filtered(data, lowcut_position, highcut_position, lowcut_signal, highcut_signal, fig, ax):
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


def position_cut(data, dataset_name, figsize=None):
    if figsize is None:
        figsize = (8, 4)
    fig, ax = plt.subplots(figsize=figsize)
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


def optimizing_delay(iteration_step=None, delay=None, error=None, figsize=None):
    if figsize is None:
        figsize = (8, 12)
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=figsize)
    delay = np.array(delay).ravel()

    ax = axs[0]
    ax.plot(np.arange(1, len(delay) + 1), delay, ".-", color="tab:blue")
    ax.grid(True)
    ax.set_ylabel("Delay (timesteps)")
    ax.set_title("Optimizing delay between position- and THz-signal")

    ax = axs[1]
    ax.semilogy(np.arange(1, len(error) + 1), np.abs(error), ".-", color="tab:orange")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Integrated error (a.u.)")

    ax = axs[2]
    sorted_error = np.abs(error)[np.argsort(delay)]
    sorted_delay = delay[np.argsort(delay)]
    ax.plot(sorted_delay, sorted_error, ".-", color="tab:red")
    ax.grid(True)
    ax.set_xlabel("Delay (timesamples)")
    ax.set_ylabel("Integrated error (a.u.)")
    ax.set_title(f"Median distance between delays: {np.median(np.diff(sorted_delay)):.1f}")
    plt.tight_layout()
    plt.show(block=False)
    return fig, axs


def with_delay_compensation(data, interpolated_delay, consider_all_traces, dataset_name, figsize=None):
    if figsize is None:
        figsize = (8, 8)
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)
    # Without delay compensation, use temporary stored data when there was no delay compensation applied
    split_pos = data["temp"]["split_pos"]
    split_sig = data["temp"]["split_sig"]
    new_position = data["temp"]["new_position"]
    # Housekeeping, remove temporary data from "data"
    del data["temp"]

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
    ax[0].legend(loc="upper left")
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
    error_no_delay = np.copy(np.nanstd(signal_matrix, axis=1))

    # With newly found delay compensation, using the corrected dataset
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
    ax[1].legend(loc="upper left")

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

    min_error_overall = np.min(np.concatenate((error_no_delay, np.nanstd(signal_matrix, axis=1))))
    ax[2].fill_between(data["light_time"], error_no_delay,
                       0.1 * min_error_overall,
                       color="black",
                       alpha=0.8,
                       label="Delay = 0 Sa")
    ax[2].fill_between(data["light_time"],
                       np.nanstd(signal_matrix, axis=1),
                       0.1 * min_error_overall,
                       color="tab:green",
                       alpha=0.8,
                       label=f"Delay = {data['delay_value']:.3f} Sa")
    ax[2].set_yscale("log")
    ax[2].set_ylim(bottom=min_error_overall)
    ax[2].legend(loc="upper left")
    ax[2].grid(True, which="both", alpha=0.3)
    ax[2].set_axisbelow(True)
    ax[2].yaxis.set_major_formatter(EngFormatter("V"))
    if dataset_name is None:
        ax[2].set_title(f"Standard deviation")
    else:
        ax[2].set_title(f"Standard deviation of dataset {dataset_name}")
    ax[2].set_xlabel("Light time")
    plt.tight_layout()
    plt.show(block=False)
    return fig, ax


def analysis_amplitude_jitter(data, figsize=None):
    if figsize is None:
        figsize = (8, 8)
    # data is already the subset of the light-dataset
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize)
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
    zero_crossing = np.zeros(data["single_traces"].shape[1])
    zero_crossing[:] = np.nan
    for i in range(data["single_traces"].shape[1]):
        idx_min = np.min([np.argmin(data["single_traces"][:, i]), np.argmax(data["single_traces"][:, i])])
        idx_max = np.max([np.argmin(data["single_traces"][:, i]), np.argmax(data["single_traces"][:, i])])
        # Use (int) index as the basis of the x-axis for creating a linear interpolation instead of light time as
        # x-axis, to not run into trouble with interpolation of small numbers on the order of 1e-12, which is used in
        # light time.
        x = np.arange(idx_min, idx_max)
        if not np.all(np.diff(x) > 0):
            raise ValueError("The index array needs to be monotonically increasing, otherwise np.interp will not work.")
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

    ax.hist(np.max(data["single_traces"], axis=0), density=False, rwidth=0.9, color="tab:blue", bins="auto")
    ax.xaxis.set_major_formatter(EngFormatter("V"))
    ax.set_xlabel("Peak amplitude")
    ax.set_ylabel("Frequency")
    if np.abs(sigma / mu) > 3e-2:
        places = 0
    elif np.abs(sigma / mu) > 3e-3:
        places = 1
    else:
        places = 2
    ax.set_title(r"Standard deviation: $\sigma=$" + f"{sigma / mu:.{places}%}")

    ax = axs[1, 1]
    # Plot zero-crossing as a histogram
    (mu, sigma) = norm.fit(zero_crossing)
    ax.hist(zero_crossing, density=False, rwidth=0.9, color="tab:orange", bins="auto")
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.set_xlabel("Zero-crossing / jitter")
    ax.set_ylabel("Frequency")
    if sigma > 3e-15:
        places = 0
    else:
        places = 1
    ax.set_title(r"Standard deviation: $\sigma=$" + f"{EngFormatter('s', places=places)(sigma)}")

    plt.tight_layout()
    plt.show(block=False)
    return fig, ax
