# Load

This page discusses aspects of parrot's `Load` class.

You can also search for specific functions in this class (or other classes) using the search box on the top right.

## Objective of the class

It is important to note, that this class is **not** necessary to use parrot successfully.

The goal is always to extract the stored raw data and bring it into the common format used throughout parrot:

A Python dictionary containing time, delay, and signal.

It works more like a shortcut/convenience class for established recording devices and file types we typically use in our
lab

Currently implemented in the file `recording_device.py` are:

1. Lock-In amplifier (Zurich Instruments, saved as `.h5`)

2. Oscilloscope (Rohde & Schwarz, saved as `.csv`)

3. Digital oscilloscope (PicoScope, saved as `.mat`)

4. DAQ (Dewesoft, saved as `.h5`)

Special functions are implemented for the lock-in amplifier (LIA),
Ideally, two signals are recorded from the LIA: `X` and `Y`. parrot can optimize the phase angle after recording to
maximize the `X` component and minimize the `Y` component, which is then discarded.

parrot can also optimize the phase when the LIA was not the direct data acquisition (DAQ) device, but was only connected
in-between,
e.g. the two-channel signal was recorded with the Dewesoft Sirius Mini DAQ.

