<br/>
<p align="center">
  <h3 align="center">PTuner: Tuning Kernels for Performance Portability</h3>

  <p align="center">
    A OpenTuner-based Tuning Framework for Determining Performance Portable BLAS Kernels.
    <br/>
    <br/>
  </p>
</p>


## Table Of Contents

- [Table Of Contents](#table-of-contents)
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started (legacy instructions)](#getting-started-legacy-instructions)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Getting Started (current instructions)](#getting-started-current-instructions)
- [Usage](#usage)
- [Setting up the Configuration File](#setting-up-the-configuration-file)
  - [PortabilityTune Arguments](#portabilitytune-arguments)
  - [Baseline Arguments](#baseline-arguments)
  - [Define Tuner Configuration](#define-tuner-configuration)
  - [Define Output(s)](#define-outputs)
- [Sample Dataset](#sample-dataset)
  - [Supported Devices](#supported-devices)
  - [Supported Kernels \& Inputs](#supported-kernels--inputs)
- [Files](#files)
- [Roadmap](#roadmap)
- [Authors](#authors)

## About The Project

PTuner is an implementation of the Portability Tuning framework that can automatically identify high performing combinations
of kernels across a subspace of devices and inputs under
various performance objectives.

## Built With

PTuner is built on [OpenTuner](github.com/jansel/opentuner), an extensible framework for multi-objective autotuners. 

## Getting Started (legacy instructions)

To get a local copy up and running follow these simple example steps.

### Prerequisites

- Podman v3 or higher.
- A running podman machine. Note that the image will need a BLAS library to support numpy, so some machines (eg, alpine) may not work without configuring a BLAS library.

for example:
``` 
podman machine init 
podman machine start
```

### Installation

MacOS and Linux:
```
bash podmanize.sh
```
This will install the podman image.

## Getting Started (current instructions)

The podman image uses `main.py` as an entrypoint (but you will probably want to run `run.py` as well). Installing from pip is preferred.

Gather dependencies: `pip install -r requirements.txt`

## Usage

Run:
```
bash run.sh <path/to/config/file> <path/to/output/file>
```

This script takes two arguments:
- a configuration file (eg, config.yml)
- an output file (eg, graphs.pdf)
  - The outputted graphs will be in pdf format.

The run script will run PortabilityTune() tunings as defined in the configuration file, and write the requested graph(s) to the output file.

## Setting up the Configuration File

### PortabilityTune Arguments

These arguments set the PortabilityTune() arguments. They select which subset of known devices, inputs, arguments, kernel, and number of outputs should be collected.

- **devices**: A list of devices. Devices should be in the form `[CLBLAST_DEVICE_NAME, DEVICE_CLOCK_SPEED]`. 
  - Some devices have overlapping CLBlast device names, so clock speed is included to better differentiate between devices
- **inputs**: A list of inputs. Inputs should comma-separated and in the form `[m, n, k]`, `[m, n]`, or `[n]`, where `m-n-k` corresponds to the input size to tune for.
- **arguments**: A list of arguments. Arguments should be comma-separated and in the form `[argument_name, [arg_val1, arg_val2]]`. This will restrict the tunings to where *argument_name* is one of *(arg_val1, argval2, ...)*
- **kernel**: `"KERNEL_FAMILY"`
  - equivalent to the `kernel_family` tag in the CLBlast database
  - see the *supported kernels* table below for a list of supported kernels
- **num_kernels**: A list of integers; the number of output kernel configurations to select. Eg, `[1,2,3]`.
- **tuning_metric**: A cost function that will be minimized during the tuning.
  - Currently supported metrics are `geometric_mean`, `mannwhitneyu` and `performance_portability`, with shorthands `geomean`, `mwu`, and `perfport`, respectively
  - To add a metric, edit `metrics.py` and create a new TuningMetric() subclass

### Baseline Arguments

Baseline arguments are used to create "baseline" performance in output graphs. The baseline appears in each graph format;

- **baseline_inputs**: A list of inputs in the same form as **inputs** above.
  - The baseline will be evaluated for these inputs only.
- **baseline_arguments**: A list of arguments in the same form as **arguments** above.
  - The baseline will be evaluated for these arguments only.

### Define Tuner Configuration

These settings influence the Tuner's execution.

- **tuning_time**: An integer; the time, in seconds, for the tuner to tune.
- **database_path**: A string; the path to the CLBlast database to query from.
  - A sample database (see [Sample Dataset](#sample-dataset) below) is given in [`database.json`](/database/database.json).
- **oracle_adjust**: A boolean; `True` if runtime results should be adjusted relative to oracle (runtime/best_time) for each database entry, `False` otherwise.
- **minimum_coverage**: A float; the proportion of entries that a tuning must cover. Can be used to ensure sufficient coverage in the case of incomplete data.
- **parallel_evals**: An integer; currently unused.

### Define Output(s)
- **graph_types**: A list of strings representing the graphs to generate. Currently supported are `portable` and `specific`.
  - `portable`: tuning queries and evaluations are performed across all queried devices
  - `specific`: tuning queries and evaluations are performed specific to each queried device
- **graph_titles**: A list of strings; must be the same length as *graph_types*. The titles for the output graphs.
- **graph_labels**: A list of strings; must be the same length as *graph_types*. The y-axis labels for the output graphs.

## Sample Dataset

A sample database is given in [`database.json`](/database/database.json). The devices, kernels, and inputs of the dataset are shown below.

### Supported Devices

| Device Name | CLBlast Device Name | Clock Speed |
| ---- | -- | --- |
| Intel HD Graphics 500 | Intel(R) Gen9 HD Graphics NEO | 750 |
| Intel HD Graphics 580 | Intel(R) Gen9 HD Graphics NEO | 1150 |
| ARM Mali-G71 | Mali-G71 | 5 |
| NVIDIA Quadro P5000 | Quadro P5000 | 1733 |
| AMD Radeon RX Vega 64 | Radeon RX Vega | 1630 |

### Supported Kernels & Inputs

| M,N,K | Values | Kernels |
| ---- | -- | --- |
| N | 4194304 | Xaxpy, Xdot |
| M,N |512, 2048, 4096, 8192 | Invert, Copy, Pad, Transpose, PadTranspose, Xger, Xgemv |
| M,N,K | 256, 512, 1024, 4096 | Xgemm, Xgemm_Direct |

Note that some device-input-kernel tuples may not be contained within the database. The [minimum coverage](#define-tuner-configuration) parameter can be modified to ensure the tuning covers an appropriate number of device-input-kernel tuples.

## Files

- `graphs/`: folder for storing graphs
- `results/database.json`: folder/file for storing and caching tuning results
- `src`:
  - `db.py`: for parsing information & entries of the database
  - `graphs.py`: for generating device-specific and device-portable graphs (used in the ICS submission, see fig 2, 3)
  - `metrics.py`: cost metrics/loss summary functions (geomean, arithmetic mean, fleet tuning, etc)
  - `plot.py`: used for generating device-specific and device-portable graphs, *not* used in the ICS submission (see fig 2, 3 in ICS submission for similar graphs however).
  - `decide.py`: generates a decision tree that infers the optimal-performing portability tuned parameters from a small benchmark; also contains an experiment with symbolic regression. not used in the ICS submission, but might be relevant reference material if implementing Decision Tree-based selection of variants.
  - `main.py`: test the tuners using `plot.py`'s & **legacy functionality**
  - `run.py`: test the tuners using `graphs.py` & **more current functionality**

Usage for `main.py` and `run.py`:
  - `*.py config.yml /path/to/output/graph.pdf`

## Roadmap

Planned Features:
- Graphs showing tuning performance over time.

See the [open issues](https://github.com/rhochgraf21/ptuner/issues) for a list of known issues.

## Authors

* **Robert Hochgraf** - *Undergraduate Researcher, Rochester Institute of Technology* - [Robert Hochgraf](https://github.com/rhochgraf21/)
* **Sreepathi Pai** - *Assistant Professor, University of Rochester* - [Sreepathi Pai](https://github.com/sree314)