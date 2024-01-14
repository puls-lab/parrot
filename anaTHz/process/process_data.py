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
        return data

    def thz_only(self, light, **kwargs):
        raw_data = {"light": {"time": light["time"],
                              "position": light["position"],
                              "signal": light["signal"]}}
        data = {"light": PrepareData(raw_data["light"], self.recording_type, **kwargs).run()}
        CutData(data["light"]).run()
        return data

    def dark_only(self, dark, **kwargs):
        raw_data = {"light": {"time": dark["time"],
                              "position": dark["position"],
                              "signal": dark["signal"]}}
        data = {"dark": PrepareData(raw_data["dark"], self.recording_type, **kwargs).run()}
        CutData(data["dark"]).run()
        return data
