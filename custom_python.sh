#!/bin/bash

python="/homes/gcasari/bigbrain/InvSR/invsr/bin/python"

if [[ $1 == "--version" ]] || [[ $1 == "-V" ]]; then
    $python "$1"
elif [[ $@ == *"generator3.py"* ]] || [[ $@ == *"import socket"* ]] || [[ $@ == *"packaging_tool.py"* ]]; then
    $python "$@"
else
    /usr/bin/srun -Q --immediate=10 --partition=all_serial --nodelist=ailb-login-03 --account=bolelli_synthetic $python "$@"
    #/usr/bin/srun -Q --immediate=10 --partition=all_serial --nodelist=ailb-login-03 --gres=gpu:1 --account=bolelli_synthetic $python "$@"
    
    #/usr/bin/srun -Q --immediate=10 --partition=all_usr_prod --account=bolelli_synthetic --gres=gpu:1 --time 15:00 --constrain="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G" $python "$@"
    #/usr/bin/srun -Q --immediate=10 --partition=all_usr_prod --account=bolelli_synthetic --gres=gpu:1 --time 2:00:00 --constrain="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G" $python "$@"
fi
