from load import Load
from recording_device import LockInAmplifier, DewesoftDAQ, Oscilloscope, Picoscope
# Python libraries
import numpy as np
import h5py
import scipy.io as sio
from scipy.constants import c
import pickle


class RecordingType(Load):
    def __init__(self, file_name):
        super().__init__(file_name)

    def load_HDF(self,
                 lockin_delay_ch="sample.auxin0.avg",
                 lockin_x_ch="sample.x.avg",
                 lockin_y_ch="sample.y.avg", ):
        with h5py.File(self.file_name, "r") as f:
            if self.recording_device == "lockin":
                if self.recording_type == "step_and_settle":
                    self.delay = f["Delay"][:]
                    self.delay = 2 * (self.delay - np.min(self.delay)) / c
                    sig = f["Signal"][:]
                    lia = LockInAmplifier(lockin_delay_ch=None, lockin_x_ch=None, lockin_y_ch=None)
                    ang = lia._minimize_y_lockin(sig[:, 0], sig[:, 1])
                    X = sig[:, 0] * np.cos(-ang) - sig[:, 1] * np.sin(-ang)
                    # Y = sig[:,0] * np.sin(-ang) + sig[:,1] * np.cos(-ang)
                    self.signal = X
                elif self.recording_type == "single_cycle" or self.recording_type == "multi_cycle":
                    # A single_cycle/multi_cycle trace in one history/file from the Lock-In.
                    lia = LockInAmplifier(lockin_delay_ch=lockin_delay_ch,
                                          lockin_x_ch=lockin_x_ch,
                                          lockin_y_ch=lockin_y_ch)
                    lia._read_lockin_attributes(f)
                    self.time, self.delay, self.signal = lia.extract_lockin_data(f)
                else:
                    raise NotImplementedError(
                        "This recording type is currently not implemented for data recorded with the lock-in "
                        "amplifier in HDF5 format.")

            elif self.recording_device == "dewesoft":
                """Recording with Dewesoft Sirius DAQ"""
                try:
                    sample_rate = int(float(f.attrs.get("Sample_rate")))
                except TypeError:
                    sample_rate = int(float(f.attrs.get("Sample rate")))
                self.delay = f[self.dewesoft_delay_ch][:]
                if self.only_x:
                    self.signal = f[self.dewesoft_signal_ch][:]
                    self.time = np.linspace(0, len(self.signal) / sample_rate - 1 / sample_rate,
                                            len(self.signal))
                else:
                    x = f[self.dewesoft_signal_X][:]
                    y = f[self.dewesoft_signal_Y][:]
                    ang = self._minimize_y_lockin(x, y)
                    self.signal = x * np.cos(-ang) - y * np.sin(-ang)
                    self.time = np.linspace(0, len(self.signal) / sample_rate - 1 / sample_rate, len(self.signal))

            elif self.recording_device == "oscilloscope":
                """Classic recording with Oscilloscope
                Either the time signal is saved as its own vector
                or we just have starting point, step size and length."""
                try:
                    self.time = f["Time"][:]
                    self.delay = f["Delay"][:]
                    self.signal = f["Signal"][:]
                except:
                    self.delay = f["Delay"][:]
                    self.signal = f["Signal"][:]
                    try:
                        self.time = np.linspace(f['Tstart'].flat[0],
                                                f['Tstart'].flat[0] + f['Tinterval'].flat[0] * (
                                                        f['Length'].flat[0] - 1),
                                                f['Length'].flat[0])
                    except:
                        self.time = np.linspace(f.attrs['Tstart'],
                                                f.attrs['Tstart'] + f.attrs['Tinterval'] * (f.attrs['TLength'] - 1),
                                                f.attrs['TLength'])
            else:
                raise NotImplementedError(
                    "The specified recording device is currently not implemented for data recorded in HDF5 format.")

    def load_TXT(self):
        """.txt could be either Franks format or the format of the oscilloscope, try both.
                    If the file is recorded from the oscilloscope directly to USB, it is saved in .csv format
                    with 3 columns: time, delay, signal"""
        if self.file_name.split(".")[-1].lower() == "txt" or "csv":
            try:
                self.time, self.delay, self.signal = np.loadtxt(self.file_name, delimiter=",", unpack=True,
                                                                skiprows=1)
            except:
                pass
        if self.lock_in_txt:
            # Denizhans file format from oscilloscope with lock-in X and Y
            self.time, self.delay, signal_x, signal_y = np.loadtxt(self.file_name, delimiter=",", unpack=True,
                                                                   skiprows=1)
            ang = self._minimize_y_lockin(signal_x, signal_y)
            self.signal = signal_x * np.cos(-ang) - signal_y * np.sin(-ang)
            # Y = signal_x * np.sin(-ang) + signal_y * np.cos(-ang)

        """Check if its Franks fileformat with 4 columns (time_delay, delay, time_signal, signal) 
        and the first row the shaker amplitude."""
        if self.file_name.split(".")[-1].lower() == "dat" or "csv":
            osci_time_d, self.delay, osci_time_a, self.signal = np.loadtxt(self.file_name, skiprows=2, unpack=True)
            if np.array_equal(osci_time_d, osci_time_a):
                self.time = osci_time_d
                del osci_time_a, osci_time_d
            else:
                print("Error: Timebase for Delay and Signal are different, "
                      "even though they should be sampled synchronously.")

    def load_MAT(self):
        mat_contents = sio.loadmat(self.file_name)
        start = mat_contents['Tstart'][0][0]
        step = mat_contents['Tinterval'][0][0]
        stop = start + mat_contents['Length'][0][0] * step
        self.time = np.arange(start, stop, step)
        self.delay = np.ravel(mat_contents['A'])
        self.signal = np.ravel(mat_contents['B'])

    def load_PICKLE(self):
        with open(self.file_name, 'rb') as fp:
            values = pickle.load(fp)
        if np.array_equal(values['1']['x'], values['2']['x']):
            self.time = values['1']['x']
        else:
            print("Error: Timebase for Delay and Signal are different, "
                  "even though they should be sampled synchronously.")
        self.delay = values['1']['y']
        self.signal = values['2']['y']
