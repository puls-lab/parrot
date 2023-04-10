import load.load
import process.process_data
import pandas as pd

load_obj = load.load.Load(
    file_name=r"\example_data\dark.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
time, position, signal = load_obj.run()
dark_df = pd.DataFrame(data={"time": time,
                             "position": position,
                             "signal": signal})
load_obj = load.load.Load(
    file_name=r"\example_data\light.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
time, position, signal = load_obj.run()
light_df = pd.DataFrame(data={"time": time,
                              "position": position,
                              "signal": signal})
process_obj = process.process_data.Process()
data = process_obj.dark_and_thz(light_df, dark_df, scale=15e-12 / 20)
print(data.keys())
