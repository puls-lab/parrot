import h5py
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class Load:
    def __init__(self,
                 file_name,
                 recording_device="LockInAmplifier",
                 recording_type="multi_cycle",
                 debug=False):
        self.file_name = file_name
        self.debug = debug
        self.file_type = None
        self.time = None
        self.delay = None
        self.signal = None
        self.recording_device = None
        self.recording_type = None

        allowed_recording_devices = ["LockInAmplifier", "DewesoftDAQ", "Oscilloscope", "Picoscope"]
        allowed_file_types = ["HDF5", "TXT", "MAT", "PICKLE"]
        allowed_recording_types = ["single_cycle", "multi_cycle"]  # TODO: "velocity", "step_and_settle"
        if recording_device in allowed_recording_devices:
            self.recording_device = recording_device
        else:
            raise NameError(f"Give correct recording device, pick one of those:\n{allowed_recording_devices}")

        if recording_type in allowed_recording_types:
            self.recording_type = recording_type
        else:
            raise NameError(f"Give correct recording type, pick one of those:\n{allowed_recording_types}")

        self.detect_file_type()

    def run(self,
            dewesoft_delay_ch="AI 1",
            dewesoft_signal_ch="AI 2",
            dewesoft_signal_X="AI 2",
            dewesoft_signal_Y="AI 3",
            only_x=True):
        with h5py.File(self.file_name, "r") as f:
            if self.recording_device == "DewesoftDAQ":
                """Recording with Dewesoft Sirius DAQ"""
                try:
                    sample_rate = int(float(f.attrs.get("Sample_rate")))
                except TypeError:
                    sample_rate = int(float(f.attrs.get("Sample rate")))
                self.delay = f[dewesoft_delay_ch][:]
                if only_x:
                    self.signal = f[dewesoft_signal_ch][:]
                    self.time = np.linspace(0, len(self.signal) / sample_rate - 1 / sample_rate,
                                            len(self.signal))
                else:
                    x = f[dewesoft_signal_X][:]
                    y = f[dewesoft_signal_Y][:]
                    ang = self._minimize_y_lockin(x, y)
                    self.signal = x * np.cos(-ang) - y * np.sin(-ang)
                    self.time = np.linspace(0, len(self.signal) / sample_rate - 1 / sample_rate, len(self.signal))
        return {"time": self.time, "position": self.delay, "signal": self.signal}

    def _minimize_y_lockin(self, signal_x, signal_y):
        x0 = [0]
        init_simplex = np.array([0, np.deg2rad(10)]).reshape(2, 1)
        xatol = 0.01
        if self.debug:
            fig, self.ax = plt.subplots(nrows=1, ncols=2, sharex=True)
            self.ax[0].set_title("Phase between X and Y")
            self.ax[0].set_xlabel("Iteration")
            self.ax[0].set_ylabel("Phase [°]")
            self.ax[1].set_title("Y Peak-Peak")
            self.ax[1].set_xlabel("Iteration")
            self.ax[1].set_ylabel("Y_peak-peak [V]")
            self.ax[0].grid()
            self.ax[1].grid()
            fig.suptitle(f"Optimizing phase, minimizing Y component")
            self.iteration = 0
        res = minimize(self._find_optimal_phase, x0, args=(signal_x, signal_y), method="Nelder-Mead",
                       options={"disp": False,
                                "maxiter": 30,
                                "xatol": xatol,
                                "initial_simplex": init_simplex})
        return res.x[0]

    def _find_optimal_phase(self, ang, x, y):
        Y = x * np.sin(-ang) + y * np.cos(-ang)
        if self.debug:
            self.ax[0].scatter(self.iteration, np.rad2deg(ang[0]), color="tab:blue")
            self.ax[1].scatter(self.iteration, np.nanmax(Y) - np.nanmin(Y), color="tab:blue")
            self.iteration = self.iteration + 1
            print(f"Phase: {np.rad2deg(ang[0]):.1f}\tY_p-p = {np.nanmax(Y) - np.nanmin(Y):E}")
        return np.nanmax(Y) - np.nanmin(Y)

    def detect_file_type(self):
        """If the file_type is not given, try to detect the filetype from file extension and set accordingly."""
        if (self.file_name.split(".")[-1].lower() == "h5") \
                or (self.file_name.split(".")[-1].lower() == "hdf5") \
                or (self.file_name.split(".")[-1].lower() == "hdf"):
            self.file_type = "HDF5"
        elif (self.file_name.split(".")[-1].lower() == "txt") \
                or (self.file_name.split(".")[-1].lower() == "dat") \
                or (self.file_name.split(".")[-1].lower() == "csv"):
            self.file_type = "TXT"
        elif (self.file_name.split(".")[-1].lower() == "mat") \
                or (self.file_name.split(".")[-1].lower() == "m"):
            self.file_type = "MAT"
        elif self.file_name.split(".")[-1].lower() == "p":
            self.file_type = "PICKLE"
        else:
            raise NotImplementedError("Could not detect file_type.")
