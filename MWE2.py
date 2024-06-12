import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft

light_time = np.loadtxt("light_time.csv", delimiter=",")
dark = np.loadtxt("dark_single_traces.csv", delimiter=",")
light = np.loadtxt("light_single_traces.csv", delimiter=",")


# Fig.1

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(26 / 2.54, 12 / 2.54))
ax = axs[0]
ax.plot(light_time, np.mean(dark, axis=1), color="black", alpha=0.8, label="Dark")
ax.plot(light_time, np.mean(light, axis=1), color="tab:orange", label="Light")
ax.grid(True)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.xaxis.set_major_formatter(EngFormatter("s"))
ax.yaxis.set_major_formatter(EngFormatter("V"))
ax.legend()

_, dark_fft = calc_fft(light_time, np.mean(dark, axis=1))
frequency, light_fft = calc_fft(light_time, np.mean(light, axis=1))

min_THz_frequency = 0.1e12
max_THz_frequency = 10.0e12
filter_frequency = (frequency >= min_THz_frequency) & (frequency <= max_THz_frequency)

ax = axs[1]
ax.semilogy(frequency[filter_frequency], np.abs(dark_fft[filter_frequency]) ** 2, color="black", alpha=0.8,
            label="Dark")
ax.semilogy(frequency[filter_frequency], np.abs(light_fft[filter_frequency]) ** 2, color="tab:orange", label="Light")
ax.grid(True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum |V²|")
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
ax.legend()
plt.tight_layout()

# Fig.2

fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(26 / 2.54, 12 / 2.54))
ax = axs2[1]
max_amplitude = np.max(np.abs(light_fft))
gain = max_amplitude / np.abs(dark_fft)

ax.semilogy(frequency[filter_frequency], (gain[filter_frequency] * np.abs(dark_fft[filter_frequency])) ** 2,
            color="black", alpha=0.8,
            label=r"Dark $\cdot$ inverse gain of dark")
ax.semilogy(frequency[filter_frequency], 1e8 * (np.abs(light_fft[filter_frequency])) ** 2, color="tab:blue",
            label=r"Light $\cdot 10^8$")
ax.semilogy(frequency[filter_frequency], (gain[filter_frequency] * np.abs(light_fft[filter_frequency])) ** 2,
            color="tab:orange", label=r"Light $\cdot$ inverse gain of dark")
ax.grid(True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum |V²|")
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
ax.legend(loc="upper right")

ax = axs2[0]

dark_irfft = np.fft.irfft(gain * dark_fft)
light_ifft = np.fft.irfft(gain * light_fft)

ax.plot(light_time, dark_irfft, color="black", alpha=0.8, label="Dark")
ax.plot(light_time, light_ifft, color="tab:orange", label="Light")
ax.grid(True)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.xaxis.set_major_formatter(EngFormatter("s"))
ax.yaxis.set_major_formatter(EngFormatter("V"))
ax.legend()
plt.tight_layout()

# Fig.3

fig3, axs3 = plt.subplots(figsize=(26 / 2.54, 12 / 2.54))
dark_fft_single_traces = np.fft.rfft(dark, axis=0).T
dark_fft_smooth = np.mean(np.abs(dark_fft_single_traces.T), axis=1)

light_fft_single_traces = np.fft.rfft(light, axis=0).T
light_fft_smooth = np.mean(np.abs(light_fft_single_traces.T), axis=1)

ax = axs3
ax.semilogy(frequency, np.abs(dark_fft) ** 2, color="black", alpha=0.8, label="Dark, average in time domain")
ax.semilogy(frequency, np.abs(dark_fft_smooth) ** 2, color="grey", alpha=0.8,
            label="Dark, absolute average in freq domain")

ax.semilogy(frequency, np.abs(light_fft) ** 2, color="tab:orange", alpha=0.8, label="Light, average in time domain")
ax.semilogy(frequency, np.abs(light_fft_smooth) ** 2, color="tab:blue", alpha=0.8,
            label="Light, absolute average in freq domain")

ax.grid(True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum |V²|")
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
ax.legend(loc="upper right")
plt.tight_layout()

# Fig.4

fig4, axs4 = plt.subplots(figsize=(26 / 2.54, 12 / 2.54))
ax = axs4

smooth_gain = np.max(dark_fft_smooth) / dark_fft_smooth

ax.semilogy(frequency, (smooth_gain * np.abs(dark_fft)) ** 2, color="black", alpha=0.8,
            label=r"Dark, avg. in time domain $\cdot$ smooth gain")
ax.semilogy(frequency, (smooth_gain * np.abs(dark_fft_smooth)) ** 2, color="grey", alpha=0.8,
            label=r"Dark, abs. avg. in freq domain $\cdot$ smooth gain")

ax.semilogy(frequency, (smooth_gain * np.abs(light_fft)) ** 2, color="tab:orange", alpha=0.8,
            label=r"Light, avg. in time domain $\cdot$ smooth gain")
# ax.semilogy(frequency, (smooth_gain * np.abs(light_fft_smooth))**2, color="tab:blue", alpha=0.8, label="Light, absolute average in freq domain")

ax.grid(True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum |V²|")
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
ax.legend(loc="upper right")
plt.tight_layout()

# Fig. 4b)

fig4, axs4 = plt.subplots(figsize=(26 / 2.54, 12 / 2.54))
ax = axs4

smooth_gain = np.max(dark_fft_smooth) / dark_fft_smooth

dark_fft_smooth_complex =  np.mean(dark_fft_single_traces.T.real, axis=1) + np.mean(dark_fft_single_traces.T.imag, axis=1)

ax.semilogy(frequency, np.abs(dark_fft / dark_fft_smooth_complex) ** 2, color="black", alpha=0.8,
            label=r"Dark, avg. in time domain / smooth gain")
ax.semilogy(frequency, np.abs(dark_fft_smooth / dark_fft_smooth_complex) ** 2, color="grey", alpha=0.8,
            label=r"Dark, abs. avg. in freq domain / smooth gain")

ax.semilogy(frequency, np.abs(light_fft / dark_fft_smooth_complex) ** 2, color="tab:orange", alpha=0.8,
            label=r"Light, avg. in time domain / smooth gain")
ax.semilogy(frequency, np.abs(light_fft_smooth / dark_fft_smooth_complex)**2, color="tab:blue", alpha=0.8, label="Light, abs. avg. in freq domain / smooth gain")

ax.grid(True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum |V²|")
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
ax.legend(loc="upper right")
plt.tight_layout()

# Fig. 5

fig5, axs5 = plt.subplots(nrows=1, ncols=2, figsize=(26 / 2.54, 12 / 2.54))
ax = axs5[1]
ax.semilogy(frequency[filter_frequency], 1e3 * (np.abs(dark_fft[filter_frequency])) ** 2, color="black", alpha=0.8,
            label=r"Dark $\cdot$ 1e3")
ax.semilogy(frequency[filter_frequency], (smooth_gain[filter_frequency] * np.abs(dark_fft[filter_frequency])) ** 2,
            color="grey", alpha=0.8,
            label=r"Dark $\cdot$ smooth gain")

ax.semilogy(frequency[filter_frequency], 1e3 * (np.abs(light_fft[filter_frequency])) ** 2, color="tab:blue",
            label=r"Light $\cdot$ 1e3")
ax.semilogy(frequency[filter_frequency], (smooth_gain[filter_frequency] * np.abs(light_fft[filter_frequency])) ** 2,
            color="tab:orange", label=r"Light $\cdot$ smooth gain")
ax.grid(True)
ax.set_xlabel("Frequency")
ax.set_ylabel("Power spectrum |V²|")
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
ax.legend(loc="upper right")

ax = axs5[0]

dark_irfft = np.fft.irfft(smooth_gain * dark_fft)
light_ifft = np.fft.irfft(smooth_gain * light_fft)

ax.plot(light_time, np.mean(dark, axis=1) / np.max(np.mean(light, axis=1)), color="black", alpha=0.8, label="Dark")
ax.plot(light_time, dark_irfft / np.max(light_ifft), color="grey", alpha=0.8, label=r"Dark $\cdot$ smooth gain")

ax.plot(light_time, np.mean(light, axis=1) / np.max(np.mean(light, axis=1)), color="tab:blue", label="Light")
ax.plot(light_time, light_ifft / np.max(light_ifft), color="tab:orange", label=r"Light $\cdot$ smooth gain")

ax.grid(True)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude (norm. to light peak)")
ax.xaxis.set_major_formatter(EngFormatter("s"))
# ax.yaxis.set_major_formatter(EngFormatter("V"))
ax.legend()
plt.tight_layout()
plt.show()
