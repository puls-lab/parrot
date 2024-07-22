# To locate the resources
from importlib import resources
# To read-in the raw data
import h5py
# To keep the plot open
import matplotlib.pyplot as plt

# Own libraries
import parrot

# Check if example files in example_data are available, otherwise download from zenodo.org
parrot.example.init.run()

my_file = (resources.files(parrot.example.example_data) / "light.h5")
with h5py.File(my_file, "r") as f:
    time = f["time"][:]
    position = f["position"][:]
    signal = f["signal"][:]
    light = {"time": time, "position": position, "signal": signal}

my_file = (resources.files(parrot.example.example_data) / "dark1.h5")
with h5py.File(my_file, "r") as f:
    time = f["time"][:]
    position = f["position"][:]
    signal = f["signal"][:]
    dark1 = {"time": time, "position": position, "signal": signal}

my_file = (resources.files(parrot.example.example_data) / "dark2.h5")
with h5py.File(my_file, "r") as f:
    time = f["time"][:]
    position = f["position"][:]
    signal = f["signal"][:]
    dark2 = {"time": time, "position": position, "signal": signal}

scale = 50e-12 / 20
data = parrot.process.thz_and_two_darks(light, dark1, dark2, scale=scale, debug=True)

data = parrot.post_process_data.correct_systematic_errors(data)
data = parrot.post_process_data.window(data)
data = parrot.post_process_data.pad_zeros(data)

parrot.plot.simple_multi_cycle(data)
parrot.plot.extended_multi_cycle(data)
plt.show(block=True)