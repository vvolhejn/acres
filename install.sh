#!/bin/bash
set -e

if python3 -c "import sys; exit(not hasattr(sys, 'real_prefix'))"; then
    echo "Please deactivate virtualenv before proceeding."
    exit 1
fi

if ! python3 -c "import virtualenv"; then
    echo "Installing virtualenv."
    python3 -m pip install virtualenv
fi

if ! [ -d env/ ]; then
    echo "Creating virtual environment"
    python3 -m virtualenv env/ -p $(which python3)
fi

source env/bin/activate
pip install -e .

echo "Don't forget to activate virtualenv: source env/bin/activate"
