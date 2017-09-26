#!/bin/bash
echo "Launching IPython Notebook from TVB Distribution"
if [ -z "$LANG" ]; then
    export LANG=en_US.UTF-8
fi
export LC_ALL=$LANG

# add tvb data and library to path and launch notebook
export PYTHONPATH=$(pwd)/tvb-data:$(pwd)/tvb-library:$PYTHONPATH;
jupyter notebook