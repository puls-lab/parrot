import load.load
import process.process_data
import plot.plot
# TODO: Delete time later
import time
from matplotlib.ticker import EngFormatter

start = time.time()
load_obj = load.load.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\anaTHz\example_data\dark.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
dark = load_obj.run()

load_obj = load.load.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\anaTHz\example_data\light.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
light = load_obj.run()
end = time.time()
print(f"TIME: Loading files: {EngFormatter('s', places=1)(end - start)}")

start = time.time()
process_obj = process.process_data.Process()
data = process_obj.dark_and_thz(light, dark, scale=15e-12 / 20)
end = time.time()
print(f"TIME: Processing files: {EngFormatter('s', places=1)(end - start)}")

plot_obj = plot.plot.Plot()
plot_obj.plot_full_multi_cycle(data)