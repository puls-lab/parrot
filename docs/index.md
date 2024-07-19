# Getting started with parrot

*parrot* is designed to process continuously recorded THz traces from a THz-TDS.
*parrot* can process the raw, continuous measurement data (consisting of time, position of the delay stage, and the THz
signal) and retrieves an averaged, high quality THz trace with access to interpolated individual traces as well as
further
statistics.

See the jupyter notebook `example.ipynb`, `example_compact.ipynb`, and python file `example.py` in the root directory,
which show step-by-step how *parrot* can be used.

To achieve this goal, parrot is divided into three main modules:

1. [Load](./load.md)

2. [Process](./process.md)

3. [Plot](./plot.md)

You can also find the corresponding documentation for each module also in the header above.

The program is released as Free and Open Source Software (FOSS), written by Tim Vogel at the Photonics and Ultrafast
Laser Science (PULS) group at the Ruhr-University Bochum, Germany.

## Virtual environment for forking/debugging of parrot
* The following steps are meant for developing purposes of :parrot: and should be not necessary to just run :parrot:.*
The requirements/enviroment is stored in `enivronment.yml`, which is automatically created by running `conda env export --from-history > environment.yml`.
Unfortunately, there is currently a bug in venv of Python, which fails to install the classic `requirement.txt` due to a cmake-error, thus the use of an anaconda enviroment.
To manually fork the code and create an associated virtual environment, you just have to run the following commands inside the root-directory of the downloaded source files:
```
conda create --name parrot --file .\environment.yml
conda activate parrot
python .\Example_compact.py
```