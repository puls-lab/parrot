import parrot

import numpy as np
import h5py
import pandas as pd
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

with h5py.File(
        r"C:\Users\Tim\PycharmProjects\anaTHzv2\parrot\example_data\Formatted_2024-05-29_Cd3As2_reprate_40.0kHz_power_30.0mW_pos_-94.00mm_120.00mm.h5",
        'r') as f:
    key_1 = list(f.keys())

    time = f[f"{key_1[0]}/time"][:]
    delay = f[f"{key_1[0]}/delay"][:]
    signal = f[f"{key_1[0]}/signal"][:]
    light = {"time": time, "position": delay, "signal": signal}


process_obj = parrot.Process()
data1 = process_obj.dark_only(dark1, scale=50e-12 / 20, delay_value=-1.526, debug=False)

post_obj = parrot.PostProcessData(data1)
#data1 = post_obj.calc_fft()
#data1 = post_obj.subtract_polynomial(order=4)
#data1 = post_obj.super_gaussian(window_width=0.6)

process_obj = parrot.Process()
data2 = process_obj.dark_only(dark2, scale=50e-12 / 20, delay_value=-1.526, debug=False)

#post_obj = parrot.PostProcessData(data2)
#data2 = post_obj.subtract_polynomial(order=4)
data1["dark"]["single_traces"] -= data2["dark"]["average"]['time_domain'].reshape(-1,1)
data1 = post_obj.super_gaussian(window_width=0.6)

process_obj = parrot.Process()
data3 = process_obj.thz_and_dark(light, dark1, scale=50e-12 / 20, delay_value=-1.526, debug=False)

post_obj = parrot.PostProcessData(data3)
data3["light"]["single_traces"] -= data3["dark"]["average"]['time_domain'].reshape(-1,1)
#data3 = post_obj.subtract_polynomial(order=4)
data3 = post_obj.super_gaussian(window_width=0.6)

# _ = post_obj.pad_zeros(new_frequency_resolution=25e9)
# data = post_obj.calc_fft()
# print(data.keys())
# print(data["light"].keys())

#data = post_obj.get_statistics()

#print(data["statistics"])

plot_obj = parrot.Plot()
#plot_obj.plot_full_multi_cycle(data, snr_timedomain=True)
dt = (data1["dark"]["light_time"][-1] - data1["dark"]["light_time"][0]) / (len(data1["dark"]["light_time"]) - 1)
frequency1 = np.fft.rfftfreq(len(data1["dark"]["light_time"]), dt)
matrix1 = np.fft.rfft(data1["dark"]["single_traces"], axis=0).T

fig, ax = plt.subplots()
ax.semilogy(frequency1, np.mean(np.abs(matrix1.T)**2, axis=1))

data1 = process_obj.dark_only(dark1, scale=50e-12 / 20, delay_value=-2, debug=False)
post_obj = parrot.PostProcessData(data1)
data1 = post_obj.subtract_polynomial(order=4)
data1 = post_obj.super_gaussian(window_width=0.6)
data1 = post_obj.calc_fft()
ax.semilogy(data1["dark"]["frequency"], np.abs(data1["dark"]["average"]["frequency_domain"])**2)

smooth_dark = np.mean(np.abs(matrix1.T)**2, axis=1)

inverse_gain = np.max(smooth_dark) / smooth_dark

#dt = (data2["dark"]["light_time"][-1] - data2["dark"]["light_time"][0]) / (len(data2["dark"]["light_time"]) - 1)
#frequency2 = np.fft.rfftfreq(len(data2["dark"]["light_time"]), dt)
#matrix2 = np.fft.rfft(data2["dark"]["single_traces"], axis=0).T
#
#ax.semilogy(frequency2, np.mean(np.abs(matrix2.T)**2, axis=1))

dt = (data3["light"]["light_time"][-1] - data3["light"]["light_time"][0]) / (len(data3["light"]["light_time"]) - 1)
frequency3 = np.fft.rfftfreq(len(data3["light"]["light_time"]), dt)
matrix3 = np.fft.rfft(data3["light"]["single_traces"], axis=0).T

ax.semilogy(frequency3, np.mean(np.abs(matrix3.T)**2, axis=1), color="tab:red")

data3 = process_obj.thz_and_dark(light, dark1, scale=50e-12 / 20, delay_value=-2, debug=False)
data3["light"]["single_traces"] -= data3["dark"]["average"]['time_domain'].reshape(-1,1)
post_obj = parrot.PostProcessData(data3)
data3 = post_obj.super_gaussian(window_width=0.6)
data3 = post_obj.calc_fft()
ax.semilogy(data3["light"]["frequency"], np.abs(data3["light"]["average"]["frequency_domain"])**2)


ax.grid(True)
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
plt.tight_layout()

fig, ax = plt.subplots()
ax.semilogy(frequency1, inverse_gain * np.mean(np.abs(matrix1.T)**2, axis=1))
ax.semilogy(data1["dark"]["frequency"], inverse_gain * np.abs(data1["dark"]["average"]["frequency_domain"])**2)
ax.semilogy(frequency3, inverse_gain * np.mean(np.abs(matrix3.T)**2, axis=1), color="tab:red")
ax.semilogy(data3["light"]["frequency"], inverse_gain * np.abs(data3["light"]["average"]["frequency_domain"])**2)
ax.grid(True)
ax.xaxis.set_major_formatter(EngFormatter("Hz"))
plt.tight_layout()
plt.show()
