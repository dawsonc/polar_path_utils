# polar_path_utils
Utilities for planning paths in polar coordinates

## Installation

It's recommended to install in a `conda` environment; create one using `conda create -n zen_table python=3.9` and activate it using `conda activate zen_table`.

If you just want to use the module, run `pip install .` from the top-level directory.

If you want to install the module to develop it, run `pip install -e .[dev]`

## Usage

`polar_path_utils` supports the following paths, each of which can be planned using a different command line interface. Run `python polar_path_utils/<script>.py --help` for usage information.

- `polar_path_utils/lines.py`: generate paths that look like straight lines.
- `polar_path_utils/spirals.py`: generate paths that look like spirals or segments of a circle.
- `polar_path_utils/lissajous.py`: generate paths that are [Lissajous figures](https://en.wikipedia.org/wiki/Lissajous_curve).

### Examples

Plot a line:
```bash
python polar_path_utils/lines.py --plot --start_pt_polar 1.0 0.0 --end_pt_polar 0.5 2 --duration 10
```

Plot a spiral:
```bash
python polar_path_utils/spirals.py --plot --start_pt_polar 1.0 0.0 --end_pt_polar 0.5 2 --duration 10
```

Plot a Lissajous figure:
```bash
python polar_path_utils/lissajous.py --plot --duration 100 --radial_frequency 3 --angular_frequency 4
```

You can save any of these figures to a file using the `--save_path path/to/save.csv`; otherwise, the default is to print the path to `stdout`.
