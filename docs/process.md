# Process

This page discusses aspects concerning the `Process` class of parrot.

## Objective of the class

Continuously recorded raw data containing several cycles of a sinusoidal delay line with the corresponding THz traces
are processed.
In particular, the possible delay difference between position and THz signal is compensated. This is possible because
the signal itself is used as a self-reference in the forward/backward motion of the delay state, which ideally should
overlap.
By correcting this phase delay between the two channels, as well as possible bandpass filtering, we can obtain a
high-quality averaged THz trace.
In addition, statistics can be computed from the large number of traces, which is particularly useful for determining
the dynamic range (DR) and signal-to-noise ratio (SNR) of the traces.

## Prepare data

As with the other classes, a logger is started which collects messages at various places in the program. There are
multiple log-levels and the standard is the log-level `WARNING`, to not flood the print-out with too many messages when
analyzing multiple files.
By setting `parrot.PrepareData(debug=True)`, the log-level is reduced to `INFO` and, as the name implies, more
information are given what the program is doing, which is in particular helpful when analyzing a single file.

Prepare data splits the process in multiple steps, which are discussed in the following:

### 1. `filter_position`

Most DAQs have the same sampling rate for the position signal as for the THz signal (and this is recommended anyway for
synchronicity reasons). However, while the THz signal requires a reasonably high sampling rate (depending on e.g. the
time window in the light time, the bandwidth of the THz pulse), commercial voice-coil shakers typically have a
sinusoidal motion profile and frequencies in the order of tens of Hz. This results in massive oversampling of the
position signal, adding high frequency noise. This is especially bad because this signal is later used for the x-axis,
resulting in fluctuating x-values and possible degradation of the THz signal when averaging multiple traces. The
solution is to apply a low pass filter to the position data. The user is free to apply either a low pass or high pass
filter to the data. In general, it is recommended to apply only a low pass filter with a -3dB frequency of 5x the
oscillation frequency. For a 20 Hz oscillating delay line, an appropriate low pass filter would be 100 Hz. This ensures
that subtle distortions that correspond to real position deviations are captured, without having high frequency noise
that is not physically caused by the movement of the stage.

Internally, the low-/high-/bandpass filter `sosfiltfilt`
from [scipy.signal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html#scipy.signal.sosfiltfilt)
are used. This filter is a forward-backward digital filter, meaning it is applied once in forward direction and once
backward direction. This results in a net-zero phase effect from the filter, which is important to not increase the
phase delay between position and THz signal even more. `sos` of `sosfiltfilt` stands for cascaded second-order sections,
since they have fewer numerical problems.

### 2. `filter_signal`

It may be useful to apply a filter to the THz signal as well. The same `sosfiltfilt` function is used.

However, it makes sense to apply different filter settings. If you want to apply a filter to the Thz data at all, a high
pass filter in the order of the shake frequency is usually a good choice. Slight misalignment of the probe beam or other
sources of noise such as vibration can affect the baseline of the THz data. In particular, the frequency of the shaker
can have a relatively small amplitude effect on the THz trace. A high-pass filter can remove any offset or low-frequency
fluctuations, making it easier to superimpose and effectively average the THz trace. A similar approach, which may be
even better for baseline correction, is to record a "dark" trace, i.e. all settings of the experiment are the same, only
the THz beam is blocked. This data file should have the same baseline as the THz signal. Later, these two traces can be
subtracted from each other.

### 3. `resample_data`

f the DAQ's sampling rate was too high for the signal bandwidth, we will introduce unnecessary high-frequency noise even
though we have no signal bandwidth at those frequencies (at lab frequencies).
We can resample the data to reduce the data rate. This will also speed up the processing of the measurement data because
downsampling reduces memory requirements.

However, resampling the data is a bit tricky because we have the sampling time in lab time but not in light time [ps].
`self.max_THz_frequency` is defined in the time frame of the THz sample. The maximum slope of the position data vs. lab
time is the smallest max. THz frequency.

```py
max_native_frequency = 1 / (np.max(np.gradient(self.data["position"], self.dt)) * self.data["scale"] * self.dt)
```

where `self.dt` is the time step of the raw data (inverse of the sampling rate), `self.data["scale"]` is the scale to
convert from the position value in [V] to the light time is [s].

*TODO: Before downsampling, it would be better to low-pass filter the data. This would not only improve the processing
speed due to a smaller memory size, but also increase the fidelity of all THz traces due to the removal of high
frequency noise.
However, it depends on the THz system in use whether the ADC is really the limiting noise factor of the system.*

### 4. `get_multiple_index`

To divide a continuous recording into subsections, each containing only a single THz trace, the extrema of the
sinusoidal delay must be detected. The function performs a Fast Fourier Transformation (FFT) and analyzes the peak in
the frequency domain that is not the DC peak. The inverse of this frequency is then used as the minimum distance to
reliably find all peaks using `find_peaks`
from [scipy.signal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

### 5. `cut_incomplete_traces`

The oscillating delay line is typically not synchronized with the data acquisition.
have a random position and are not useful for the THz averaging process. To ensure that only complete traces are
processed later, the first and last trace are removed from the data set.