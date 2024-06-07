import anaTHz as parrot

load_obj = parrot.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\anaTHz\example_data\dark.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
dark = load_obj.run()

load_obj = parrot.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\anaTHz\example_data\light.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
light = load_obj.run()

process_obj = parrot.Process()
data = process_obj.dark_and_thz(light, dark, scale=15e-12 / 20, debug=True)

post_obj = parrot.PostProcessData(data)
data = post_obj.subtract_polynomial(order=4)
data = post_obj.super_gaussian(window_width=0.6)
# _ = post_obj.pad_zeros(new_frequency_resolution=25e9)
# data = post_obj.calc_fft()
# print(data.keys())
# print(data["light"].keys())

data = post_obj.get_statistics()

print(data["statistics"])

plot_obj = parrot.Plot()
plot_obj.plot_full_multi_cycle(data, snr_timedomain=True)
# plot_obj.plot_simple_multi_cycle(data)
