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
        self.load_file(recording_type)

    def detect_file_type(self):
        """If the file_type is not given, try to detect the filetype from file extension and set accordingly."""
        if (self.file_name.split(".")[-1].lower() == "h5") \
                or (self.file_name.split(".")[-1].lower() == "hdf5") \
                or (self.file_name.split(".")[-1].lower() == "hdf"):
            self.file_type = "HDF5"
        if (self.file_name.split(".")[-1].lower() == "txt") \
                or (self.file_name.split(".")[-1].lower() == "dat") \
                or (self.file_name.split(".")[-1].lower() == "csv"):
            self.file_type = "TXT"
        if (self.file_name.split(".")[-1].lower() == "mat") \
                or (self.file_name.split(".")[-1].lower() == "m"):
            self.file_type = "MAT"
        if self.file_name.split(".")[-1].lower() == "p":
            self.file_type = "PICKLE"
        else:
            raise NotImplementedError("Could not detect file_type.")

    def load_file(self):
