import parrot
import h5py
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

load_obj = parrot.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\parrot\example_data\2024-05-29_Dark_LN_Cryo_60s_40kHz_80K_-19.7mm_1,.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
dark1 = load_obj.run()


load_obj = parrot.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\parrot\example_data\2024-05-29_Dark2_LN_Cryo_60s_40kHz_80K_-19.7mm_1,.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
dark2 = load_obj.run()

load_obj = parrot.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\parrot\example_data\light.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
light = load_obj.run()

output_file_name = r"C:\Users\Tim\PycharmProjects\anaTHzv2\parrot\example_data\Formatted_2024-05-29_Cd3As2_reprate_40.0kHz_power_30.0mW_pos_-94.00mm_120.00mm.h5"

with h5py.File(output_file_name, 'r') as f:
    key_1 = list(f.keys())

    time = f[f"{key_1[0]}/time"][:]
    delay = f[f"{key_1[0]}/delay"][:]
    signal = f[f"{key_1[0]}/signal"][:]
    light = {"time": time, "position": delay, "signal": signal}

# process_obj = parrot.Process()
# data = process_obj.thz_and_two_darks(light, dark1, dark2, delay_value=-1.526, scale=50e-12 / 20, debug=True)
# data = process_obj.thz_and_dark(light, dark1, delay_value=-1.526, scale=50e-12 / 20, debug=True)

# post_obj = parrot.PostProcessData(data)
# data = post_obj.subtract_polynomial(order=4)
# data = post_obj.correct_systematic_errors()
# data = post_obj.super_gaussian(window_width=0.6)
# _ = post_obj.pad_zeros(new_frequency_resolution=25e9)
# data = post_obj.calc_fft()
# print(data.keys())
# print(data["light"].keys())

# data = post_obj.get_statistics()

# print(data["statistics"])

# plot_obj = parrot.Plot()
# plot_obj.plot_full_multi_cycle(data, snr_timedomain=True)

start_bandwidth = 0.1e12
min_THz_frequency = 0.05e12
stop_bandwidth = 3e12
max_THz_frequency = 10e12


def calc_fft(time, signal):
    dt = (time[-1] - time[0]) / (len(time) - 1)
    frequency = np.fft.rfftfreq(len(time), dt)
    signal_fft = np.fft.rfft(signal)
    return frequency, signal_fft


process_obj = parrot.Process()
data = process_obj.thz_and_two_darks(light, dark1, dark2, delay_value=-1.526, scale=50e-12 / 20, debug=False)
post_obj = parrot.PostProcessData(data)
# data = post_obj.correct_systematic_errors()
data = post_obj.super_gaussian(window_width=0.6)

# np.savetxt("light_time.csv", data["light"]["light_time"], delimiter=",")
# np.savetxt("dark_single_traces.csv", data["dark"]["single_traces"], delimiter=",")
#np.savetxt("light_single_traces.csv", data["light"]["single_traces"], delimiter=",")

fig, ax = plt.subplots(nrows=3, ncols=2, sharex="row", figsize=(22 / 2.54, 20 / 2.54))
frequency_dark, dark_fft = calc_fft(data["dark1"]["light_time"], data["dark1"]["average"]["time_domain"])
dark_fft1 = np.copy(dark_fft)

filter_frequency = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)

ax[0, 0].plot(data["dark1"]["light_time"], data["dark1"]["average"]["time_domain"],
              color="black",
              alpha=0.8,
              label="Dark 1")

ax[1, 0].plot(data["dark1"]["light_time"], data["dark1"]["average"]["time_domain"],
              color="black",
              alpha=0.8,
              label="Dark 1")

ax[2, 0].plot(frequency_dark[filter_frequency],
              np.abs(dark_fft[filter_frequency]) ** 2,
              color="black",
              alpha=0.8,
              label="Dark 1")

frequency_dark, dark_fft = calc_fft(data["dark2"]["light_time"], data["dark2"]["average"]["time_domain"])
dark_fft2 = np.copy(dark_fft)
filter_frequency = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)
ax[0, 0].plot(data["dark2"]["light_time"], data["dark2"]["average"]["time_domain"],
              color="grey",
              alpha=0.8,
              label="Dark 2")

ax[1, 0].plot(data["dark2"]["light_time"], data["dark2"]["average"]["time_domain"],
              color="grey",
              alpha=0.8,
              label="Dark 2")

ax[2, 0].plot(frequency_dark[filter_frequency],
              np.abs(dark_fft[filter_frequency]) ** 2,
              color="grey",
              alpha=0.8,
              label="Dark 2")

frequency_light, light_fft = calc_fft(data["light"]["light_time"], data["light"]["average"]["time_domain"])
filter_frequency = (frequency_light >= min_THz_frequency) & (frequency_light <= max_THz_frequency)

frequency, signal_fft = frequency_light[filter_frequency], light_fft[filter_frequency]
ax[0, 0].plot(data["light"]["light_time"], data["light"]["average"]["time_domain"],
              color="tab:orange",
              alpha=0.8,
              label="Light")
ax[1, 0].plot(data["light"]["light_time"], data["light"]["average"]["time_domain"],
              color="tab:orange",
              alpha=0.8,
              label="Light")

ax[2, 0].semilogy(frequency, np.abs(signal_fft) ** 2,
                  color="tab:orange",
                  alpha=0.8,
                  label="Light")

fig2, ax2 = plt.subplots()

ax2.semilogy(frequency_dark[filter_frequency],
             np.abs(dark_fft1[filter_frequency]) ** 2 - np.abs(dark_fft2[filter_frequency]) ** 2,
             color="black",
             label="|FFT(Dark1)|² - |FFT(Dark2)|²")

ax2.semilogy(frequency,
             np.abs(signal_fft) ** 2 - np.abs(dark_fft1[filter_frequency]) ** 2,
             color="tab:orange",
             label="|FFT(Light)|² - |FFT(Dark1)|²")

ax2.grid(True)
ax2.legend(loc="upper right")
ax2.set_ylabel("Power spectrum (|V²|)")
ax2.xaxis.set_major_formatter(EngFormatter("Hz"))
ax2.set_xlabel("Frequency")
# plt.savefig("Fig.2.png", dpi=300)
plt.close(fig2)

fig3, ax3 = plt.subplots()

ax3.plot(frequency_dark[filter_frequency],
         10 * np.log10(np.abs(dark_fft1[filter_frequency]) ** 2) - 10 * np.log10(
             np.abs(dark_fft2[filter_frequency]) ** 2),
         color="black",
         label=r"$10\cdot \log_{10}(|\mathrm{FFT(Dark1)}|^2) - 10\cdot \log_{10}(|\mathrm{FFT(Dark2)}|^2)$")

ax3.plot(frequency,
         10 * np.log10(np.abs(signal_fft) ** 2) - 10 * np.log10(np.abs(dark_fft1[filter_frequency]) ** 2),
         color="tab:orange",
         label=r"$10\cdot \log_{10}(|\mathrm{FFT(Light)}|^2) - 10\cdot \log_{10}(|\mathrm{FFT(Dark1)}|^2)$")

ax3.grid(True)
ax3.legend(loc="upper right")
ax3.set_ylabel("Power spectrum (dB)")
ax3.xaxis.set_major_formatter(EngFormatter("Hz"))
ax3.set_xlabel("Frequency")
# plt.savefig("Fig.3.png", dpi=300)
plt.close(fig3)

process_obj = parrot.Process()
data = process_obj.thz_and_two_darks(light, dark1, dark2, delay_value=-1.526, scale=50e-12 / 20, debug=False)

post_obj = parrot.PostProcessData(data)
# data = post_obj.subtract_polynomial(order=4)
data = post_obj.correct_systematic_errors()
data = post_obj.super_gaussian(window_width=0.6)

frequency_dark, dark_fft = calc_fft(data["dark"]["light_time"], data["dark"]["average"]["time_domain"])
filter_frequency = (frequency_dark >= min_THz_frequency) & (frequency_dark <= max_THz_frequency)
ax[0, 1].plot(data["dark"]["light_time"], data["dark"]["average"]["time_domain"],
              color="black",
              alpha=0.8,
              label="Dark 1 - Dark 2")

ax[1, 1].plot(data["dark"]["light_time"], data["dark"]["average"]["time_domain"],
              color="black",
              alpha=0.8,
              label="Dark 1 - Dark 2")

ax[2, 1].plot(frequency_dark[filter_frequency],
              np.abs(dark_fft[filter_frequency]) ** 2,
              color="black",
              alpha=0.8,
              label="Dark 1 - Dark 2")

frequency_light, light_fft = calc_fft(data["light"]["light_time"], data["light"]["average"]["time_domain"])
filter_frequency = (frequency_light >= min_THz_frequency) & (frequency_light <= max_THz_frequency)

frequency, signal_fft = frequency_light[filter_frequency], light_fft[filter_frequency]
ax[0, 1].plot(data["light"]["light_time"], data["light"]["average"]["time_domain"],
              color="tab:orange",
              alpha=0.8,
              label="Light  - Dark 1")

ax[1, 1].plot(data["light"]["light_time"], data["light"]["average"]["time_domain"],
              color="tab:orange",
              alpha=0.8,
              label="Light  - Dark 1")

ax[2, 1].semilogy(frequency, np.abs(signal_fft) ** 2,
                  color="tab:orange",
                  alpha=0.8,
                  label="Light  - Dark 1")
# ax[2].semilogy(frequency_dark[filter_frequency],
#           np.abs(dark_fft[filter_frequency]) ** 2,
#           color="black",
#           alpha=0.8)
# ax[2].semilogy(frequency, np.abs(signal_fft) ** 2,
#           color="tab:orange",
#           alpha=0.8)

ax[0, 0].set_ylabel("Amplitude")
ax[0, 0].yaxis.set_major_formatter(EngFormatter("V"))
ax[0, 1].set_ylabel("Amplitude")
ax[0, 1].yaxis.set_major_formatter(EngFormatter("V"))
ax[0, 1].xaxis.set_major_formatter(EngFormatter("s"))
ax[0, 0].set_xlabel("Time")
ax[0, 1].set_xlabel("Time")

ax[1, 0].set_ylabel("Amplitude")
ax[1, 0].yaxis.set_major_formatter(EngFormatter("V"))
ax[1, 1].set_ylabel("Amplitude")
ax[1, 1].yaxis.set_major_formatter(EngFormatter("V"))
ax[1, 1].xaxis.set_major_formatter(EngFormatter("s"))
ax[1, 0].set_xlabel("Time")
ax[1, 1].set_xlabel("Time")

ax[2, 0].set_ylabel("Power spectrum (|V²|)")
ax[2, 1].set_ylabel("Power spectrum (|V²|)")
ax[2, 1].xaxis.set_major_formatter(EngFormatter("Hz"))
ax[2, 0].set_xlabel("Frequency")
ax[2, 1].set_xlabel("Frequency")

ax[1, 0].set_ylim([-5e-4, 5e-4])
ax[1, 1].set_ylim([-5e-4, 5e-4])

for axs in ax.ravel():
    axs.grid(True)
    axs.legend(loc="upper right")
plt.tight_layout()
# tmp = pd.DataFrame(data={"time" : data["light"]["light_time"],
#                         "light": data["light"]["average"]["time_domain"],
#                         "dark" : data["dark"]["average"]["time_domain"]})
# tmp.to_csv("data.csv")


# plt.savefig("Fig.1.png", dpi=300)
plt.show()
# plot_obj.plot_simple_multi_cycle(data)
