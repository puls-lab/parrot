import anaTHz.load.load
import anaTHz.process.process_data
import pandas as pd

load_obj = anaTHz.load.load.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\example\data\thzpowercorrected_20.1,probepower_24.0,.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
time, position, signal = load_obj.run()
dark_df = pd.DataFrame(data={"time": time,
                             "position": position,
                             "signal": signal})
load_obj = anaTHz.load.load.Load(
    file_name=r"C:\Users\Tim\PycharmProjects\anaTHzv2\example\data\thzpowercorrected_20.1,probepower_24.0,.h5",
    recording_device="DewesoftDAQ",
    recording_type="multi_cycle")
time, position, signal = load_obj.run()
light_df = pd.DataFrame(data={"time": time,
                              "position": position,
                              "signal": signal})
process_obj = anaTHz.process.process_data.Process()
data = process_obj.dark_and_thz(light_df, dark_df, scale=15e-12 / 20)
print(data.keys())
