"""This module contains various plot-functions which are called when debug is set to True during processing."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.optimize import brentq
from scipy.stats import norm, kde
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

    position_selection = data["position"][0 : data["trace_cut_index"][10] + 1]
    ax.plot(position_selection, label="Position")
    [ax.axvline(x, color="black", alpha=0.8) for x in data["trace_cut_index"][:11]]
    ax.axvline(np.nan, color="black", alpha=0.8, label="Cut index")
    ax.legend(loc="upper right")
    if dataset_name is None:
        ax.set_title(f"Cutting dataset into single traces")
    else:
        # If name of dataset like "light", "dark1", etc. are given, place it into the axis title.
        ax.set_title(f"Cutting {dataset_name} dataset into single traces")
    ax.set_xlabel("Time samples")
    ax.set_ylabel("Position")
    ax.set_xlim([0, data["trace_cut_index"][10] + 1])
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


def analysis(data, figsize=None):
    if figsize is None:
        figsize = (8, 12)
    fig, axs = plt.subplot_mosaic([['A', 'A', 'A2'],
                                   ['B', 'B', 'B2'],
                                   ['C', 'C', 'C2']])

    ax = axs["A"]
    # Calculate the amplitude peak of each single trace
    amplitude_peaks = np.max(data["single_traces"], axis=0)
    ax.plot(np.arange(data["number_of_traces"]), amplitude_peaks, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Trace ID")
    ax.set_ylabel("Peak amplitude")
    ax.yaxis.set_major_formatter(EngFormatter("V"))
    ax.grid(True)

    ax = axs["B"]
    # Calculate energy (in a.u.) of each single trace
    energy = np.trapz(np.abs(data["single_traces"].T) ** 2)
    ax.plot(np.arange(data["number_of_traces"]), energy, color="tab:orange", alpha=0.8)
    ax.set_xlabel("Trace ID")
    ax.set_ylabel("Energy (a.u.)")
    ax.grid(True)

    ax = axs["C"]
    # Calculate jitter based on cross-correlation
    jitter = post_process_data._get_jitter_oversampled(data)
    # Plot jitter vs. trace ID
    ax.plot(np.arange(data["number_of_traces"]), jitter, color="tab:green", alpha=0.8)
    ax.set_xlabel("Trace ID")
    ax.set_ylabel("Jitter\nvs. first trace")
    ax.yaxis.set_major_formatter(EngFormatter("s"))
    ax.grid(True)

    axs_vs_trace_id = [axs["A"], axs["B"], axs["C"]]
    for ax in axs_vs_trace_id[1:]:
        ax.sharex(axs["A"])
    for ax in axs_vs_trace_id:
        ax.set_xlim([0, data["number_of_traces"] - 1])
    fig.align_ylabels(axs_vs_trace_id)

    ax = axs["A2"]
    # Plot amplitude as a histogram / kernel density estimate (kind of like smooth histogram)
    (mu, sigma) = norm.fit(amplitude_peaks)
    density = kde.gaussian_kde(amplitude_peaks)
    if np.abs(sigma / mu) > 3e-2:
        places = 0
    elif np.abs(sigma / mu) > 3e-3:
        places = 1
    else:
        places = 2
    x = np.linspace(np.min(amplitude_peaks), np.max(amplitude_peaks), 201)
    ax.plot(x, density(x) / np.max(density(x)), color="tab:blue")
    ax.set_xlabel("Peak amplitude")
    ax.set_ylabel("Density (norm.)")
    ax.set_title(r"STD: $\sigma=$" + f"{sigma / mu:.{places}%}")
    ax.xaxis.set_major_formatter(EngFormatter("V"))
    ax.grid(True)

    ax = axs["B2"]
    # Plot energy as a histogram / kernel density estimate (kind of like smooth histogram)
    (mu, sigma) = norm.fit(energy)
    density = kde.gaussian_kde(energy)
    if np.abs(sigma / mu) > 3e-2:
        places = 0
    elif np.abs(sigma / mu) > 3e-3:
        places = 1
    else:
        places = 2
    x = np.linspace(np.min(energy), np.max(energy), 201)
    ax.plot(x, density(x) / np.max(density(x)), color="tab:orange")
    ax.set_xlabel("Energy (a.u.)")
    ax.set_ylabel("Density (norm.)")
    ax.set_title(r"STD: $\sigma=$" + f"{sigma / mu:.{places}%}")
    ax.grid(True)

    ax = axs["C2"]
    # Plot jitter as a histogram / kernel density estimate (kind of like smooth histogram)
    (mu, sigma) = norm.fit(jitter)
    density = kde.gaussian_kde(jitter)
    if sigma > 3e-15:
        places = 0
    else:
        places = 1
    ax.set_title(r"STD: $\sigma=$" + f"{EngFormatter('s', places=places)(sigma)}")
    x = np.linspace(np.min(jitter), np.max(jitter), 201)
    ax.plot(x, density(x) / np.max(density(x)), color="tab:green")
    ax.grid(True)
    ax.set_xlabel("Jitter")
    ax.set_ylabel("Density (norm.)")
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.grid(True)

    for ax in [axs["A2"], axs["B2"], axs["C2"]]:
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.show(block=False)
    return fig, ax
