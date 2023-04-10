from .prepare_data import PrepareData
from .cut_data import CutData
from .post_process_data import PostProcessData


class Process:
    def __init__(self, recording_type="multi_cycle", debug=False):
        # For testing purposes
        self.debug = debug
        self.recording_type = recording_type
        pass

    def dark_and_thz(self, df_light, df_dark, **kwargs):
        raw_data = {}
        for mode, df in zip(["light", "dark"], [df_light, df_dark]):
            raw_data[mode] = {"time": df.iloc[:, 0],
                              "position": df.iloc[:, 1],
                              "signal": df.iloc[:, 2]}
        data = {"light": PrepareData(raw_data["light"], self.recording_type, **kwargs).run(),
                "dark": {}}
        data["dark"] = PrepareData(raw_data["dark"],
                                   delay_value=data["light"]["delay_value"],
                                   **kwargs).run()
        data["light"] = CutData(data["light"])
        data["dark"] = CutData(data["dark"])
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
