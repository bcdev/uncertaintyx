# Installing

## Create the Python environment

This step is optional. If you have a running Python 3.11, 3.12, or
3.13 environment, you may continue with the next step. Otherwise,
open a terminal window, `cd` into the `uncertaintyx` directory,
and type

    conda env create --file=environment.yml

Type

    conda env list

to list the available Python environments. If the `uncertaintyx` environment
is available, the list reads like

    # conda environments:
    #
    base                  *  /...
    uncertaintyx             /.../envs/uncertaintyx

The `*` indicates the active environment. Activate the `uncertaintyx`
environment by typing

    conda activate uncertaintyx

List the available environments again

    conda env list

When the `uncertaintyx` environment is active, the list output reads like

    # conda environments:
    #
    base                     /...
    uncertaintyx          *  /.../envs/uncertaintyx

Alternatively, you may update your existing conda `base` environment

    conda env update --file environment.yml --name base

and use it instead of the `uncertaintyx` environment.

## Install the Python package

To install the Uncertaintyx Python package and its dependencies into
your Python environment type

    python -m pip install .

Repeat the installation after each update of the Uncertaintyx package.
The installation command automatically installs all missing dependencies
into your environment, too. If all dependencies are satisfied, you may
perform the installation without dependencies, i.e,

    python -m pip install --no-deps .

In a development environment, you may instead type (once)              

    python -m pip install --no-deps --editable .

which registers the current directory into your environment in a way which does
not require repeated installation.

## Run the tests

To execute unit level tests `cd` into the `uncertaintyx` directory and type

    pytest 

The `pytest` output is printed to the console.
