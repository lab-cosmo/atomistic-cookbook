#! /bin/bash

set -ex

conda_path=$(conda info --envs | grep 'cp2k_run_batch' | sed 's: ::g')

echo "all executables for cp2k_run_batch conda env"
ls $conda_path/bin

echo "List all exectubales containing 'cp2k'"
compgen -cX '!*cp2k*'
