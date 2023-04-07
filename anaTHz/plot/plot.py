class Plot:
    def __init__(self,
                 recording_type=None,
                 min_THz_frequency=0.1e12,
                 max_THz_frequency=5e12,
                 noise_floor_start_fft=4e12,
                 time_as_index=True,
                 smoothing_time=False,
                 smoothing_frequency=False,
                 plot_water_absorption=True,
                 n_largest_water_absorptions=30):

        self.recording_type = recording_type
        self.min_THz_frequency = min_THz_frequency  # In [Hz]
        self.max_THz_frequency = max_THz_frequency  # In [Hz]
        self.noise_floor_start_fft = noise_floor_start_fft  # In [Hz]
        self.time_as_index = time_as_index
        self.smoothing_time = smoothing_time  # Smooths THz trace in time-domain
        self.smoothing_time_size = 512
        self.smoothing_frequency = smoothing_frequency  # Smooths THz spectrum in freq-domain
        self.smoothing_frequency_size = 16
        # True or False to plot water absorption lines in frequency plot
        self.plot_water_absorption = plot_water_absorption
        # Reduce the amount of water absorption lines to the strongest n to not clutter the axis
        self.n_largest_water_absorptions = n_largest_water_absorptions

        self.time = None
        self.frequency = None
        self.filter_frequency = None

        self.run()

    def run(self):
        self.filter_frequency = (df["frequency"] > self.min_THz_frequency) & (df["frequency"] < self.max_THz_frequency)
        self.frequency = df.loc[self.filter_frequency, "frequency"]

        fig = None
        ax = None

        if self.recording_type == "single_cycle":
            fig, ax = self.plot_single_cycle(df)
        if self.recording_type == "multi_cycle":
            fig, ax = self.plot_multi_cycle(df)
