from .prepare_data import PrepareData
from .cut_data import CutData
from .post_process_data import PostProcessData


class Process:
    def __init__(self, recording_type="multi_cycle", debug=False):
        # For testing purposes
        self.debug = debug
        self.recording_type = recording_type
        pass

    def dark_and_thz(self, light, dark, **kwargs):
        raw_data = {}
        for mode, data in zip(["light", "dark"], [light, dark]):
            raw_data[mode] = {"time": data["time"],
                              "position": data["position"],
                              "signal": data["signal"]}
        data = {"light": PrepareData(raw_data["light"], self.recording_type, **kwargs).run(),
                "dark": {}}
        data["dark"] = PrepareData(raw_data["dark"],
                                   delay_value=data["light"]["delay_value"],
                                   **kwargs).run()
        for mode in ["light", "dark"]:
            data[mode] = CutData(data[mode]).run()
            post_obj = PostProcessData(data[mode]["light_time"])
            window = post_obj.super_gaussian()
            data[mode]["single_traces"] *= window.reshape(-1, 1)
            data[mode]["average"]["time_domain"] *= window
        return data

    def thz_only(self, light_df):
        pass
        # raw_data = {"light": {"time": df.iloc[:, 0],
        #                      "position": df.iloc[:, 1],
        #                      "signal": df.iloc[:, 2]}

    def dark_only(self, dark_df):
        pass
        # raw_data = {"dark": {"time": df.iloc[:, 0],
        #                     "position": df.iloc[:, 1],
        #                     "signal": df.iloc[:, 2]}
