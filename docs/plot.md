# Plot

This page discusses aspects of the `plot`-module of parrot.

After processing the data with the `process` module, a single Python dictionary called `data` is returned.
This dictionary contains general metadata about the dataset, such as the number of traces, but also the data itself,
specifically
each sliced & averaged individual trace, as well as averaged traces in the time and frequency domains.

The `plot` module currently supports two different functions that can work with `data`.

```python
fig, ax = parrot.plot.simple_multi_cycle(data)
fig, ax = parrot.plot.extended_multi_cycle(data)
```

Both are discussed in detail in the following sections:

## Simple multi cycle

```python
def simple_multi_cycle(data,
                       min_THz_frequency=0e12,
                       max_THz_frequency=10e12,
                       threshold_dB=10,
                       figsize=None,
                       water_absorption_lines=True,
                       debug=False):
```

Creates three subplots showing the averaged THz trace in both the time and frequency domains along with the measured
dark traces.
For the frequency domain, two perspectives are given, one on a normalized linear scale and one on a logarithmic scale.
Additional information is given in the title of the figure as well as in the plots, consisting of the signal-to-noise
ratio (SNR) in the time domain, the dynamic range in the frequency domain as well as two different bandwidths,
full-width at half-maximum (FWHM) and a bandwidth on a logarithmic scale depending on `threshold_dB`. The input
parameters of `simple_multi_cycle()` are discussed below.

***

#### `min_THz_frequency`, `max_THz_frequency`

Input: `float`.
When reading the data, the sampling rate is checked and, if `filter_signal=True` in `process.run()`, the maximum
displayable internal THz frequency is typically limited to **50 THz**, which should be sufficient for most THz
experiments. (It can, of course, be adjusted in the `process` module if an even higher THz frequency is required).

However, this large frequency is not very suitable for display purposes. For this reason, the min. and max. THz
frequency are further restricted for plotting the data. The default settings of `simple_multi_cycle()` set these to 0
THz and 10 THz.

#### `threshold_dB`

Input: `float`.
:parrot: tries to give an estimate of the received THz pulse by analyzing the full width at half maximum (FWHM) and the
bandwidth of the spectrum on a logarithmic scale. Since there is no official definition for a logarithmic bandwidth and
the
definition of the signal hitting the noise floor can often be arbitrary in time (due to the fluctuations of the noise
floor itself), a threshold of 10 dB is set as a standard parameter. This means that the largest frequency bandwidth of
the THz signal that is at least 10 dB above the noise floor is selected. Any dip in the signal or peak in the noise
floor at a given frequency will interrupt the bandwidth. Only the largest continuous portion of the THz signal above the
specified threshold is selected.

While the linear FWHM bandwidth is suitable for nonlinear THz spectroscopy (because the frequencies covered by this
bandwidth should have the relatively highest amplitude to induce nonlinear effects), the logarithmic bandwidth with a 10
dB threshold is suitable for linear THz spectroscopy. Programs such as [phoeniks](https://github.com/puls-lab/phoeniks)
offer to extract the refractive index and absorption coefficient from (averaged) THz time-domain data of a reference and
sample measurement. The algorithms are typically stable enough to extract the relative information when the signal is 10
dB above the noise floor.

#### `figsize`

Input: `None` or `(float, float)`.
This parameter is used as a shortcut to get suitable plots in the right size directly. For example, if you are
characterizing your THz source and/or detection scheme and want to tell others about it at the next meeting, you can
specify a `figsize` here that will scale the plot to the dimensions you need, e.g. for a presentation. You have to give
the width and height as a tuple and the unit is unfortunately in inches. However, you can get the appropriate dimension
in [cm] by dividing the input by 2.54.

For example:

```python
simple_multi_cycle(data, figsize=(30 / 2.54, 18 / 2.54))
plt.savefig("Plot.png", dpi=600)  # Saves the plot in the current working directory
```

#### `water_absorption_lines`

Input: `bool`.
For convenience, the water absorption lines at THz frequencies can be enabled or disabled and are extracted from the
HITRAN online database using HAPI.

!!! Reference

    R.V. Kochanov, I.E. Gordon, L.S. Rothman, P. Wcislo, C. Hill, J.S. Wilzewski,

    HITRAN Application Programming Interface (HAPI):
    A comprehensive approach to working with spectroscopic data,

    J. Quant. Spectrosc. Radiat. Transfer 177, 15-30 (2016)

    [https://hitran.org/hapi/](https://hitran.org/hapi/)

#### `debug`

Input: `bool`.
For debugging purposes.

## Extended multi cycle

```python
def extended_multi_cycle(data,
                         min_THz_frequency=0e12,
                         max_THz_frequency=10e12,
                         threshold_dB=10,
                         figsize=None,
                         snr_timedomain=False,
                         water_absorption_lines=True):
```

Creates three subplots that provide more information, particularly regarding the averaging effect of using an ensemble
of individual traces. The plot consists of the time domain trace as before, a frequency domain trace, and a plot that
shows the dynamic range for a cumulative average of the light trace versus the averaged dark trace. Ideally, this
double-log plot should show a line. Due to nonstationary noise or other drifts in the setup, the dynamic range can
become saturated or even degraded as more individual traces are averaged.

#### `snr_timedomain`

Input: `bool`. Enables/disables whether to calculate and display the signal-to-noise ratio (SNR) in time domain.

#### `min_THz_frequency`, `max_THz_frequency`, `threshold_dB`, `figsize`, `water_absorption_lines`

See description at [Simple multi cycle](#simple-multi-cycle).

