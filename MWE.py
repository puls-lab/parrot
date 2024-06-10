import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter


def calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft


_, light_time, light, dark = np.loadtxt("data.csv", delimiter=",", skiprows=1, unpack=True)

fig, axs = plt.subplots(nrows=1, ncols=2)
ax = axs[0]
ax.plot(light_time, dark, color="black", alpha=0.8, label="Dark")
ax.plot(light_time, light, color="tab:orange", label="Light")
ax.grid(True)
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
ax.xaxis.set_major_formatter(EngFormatter("s"))
ax.yaxis.set_major_formatter(EngFormatter("V"))
ax.legend()

_, dark_fft = calc_fft(light_time, dark)
frequency, light_fft = calc_fft(light_time, light)

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
plt.show()
