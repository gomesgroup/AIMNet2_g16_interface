#!/bin/bash

# An interface between Gaussian and AIMNet2 that takes care of Hessian
# calculations and command line arguments.

# if the first argument is --log-all, then the full AIMNet2 output will be
# added to the Gaussian log file
DEBUG=0
if [[ "$1" == "--log-all" ]]; then
    DEBUG=1
    shift
fi

# Final 6 arguments are those passed by Gaussian.
arg_gauss=("${@: -6}")

# First, move to the directory containing the .EIn file (the Gaussian scratch
# directory.) This is so that AIMNet2 can produce a .EOut file in the same
# directory.
Ein_dir=$(dirname "${arg_gauss[1]}")
Eout_dir=$(dirname "${arg_gauss[2]}")
Elog_dir=$(dirname "${arg_gauss[3]}")

cd "$Ein_dir" || exit

# Open input file and load parameters.
read -r natoms deriv icharg multip < "${arg_gauss[1]}"

# Setup redirection of AIMNet2 output. Here we throw it out instead, unless $DEBUG 
# is on. We do this because otherwise the Gaussian output gets way too
# cluttered.
if [[ $DEBUG -eq 1 ]]; then
    msg_output="> ${arg_gauss[3]} 2>&1"
else
    msg_output=">/dev/null 2>${arg_gauss[3]}"
fi

# Setup AIMNet2 according to run type
if [[ $deriv -lt 2 ]]; then
    runtype="--forces"
else
    runtype="--hess"
fi

aimnet2_run="python /Users/passos/GitHub/gomesgroup/AIMNet2_g16_interface/calculators/ase/aimnet2ase_exec.py --model /Users/passos/GitHub/gomesgroup/AIMNet2_g16_interface/models/aimnet2_wb97m-d3_ens.jpt --in_file ${arg_gauss[1]} --out_file ${arg_gauss[2]} $runtype --charge $icharg $msg_output"
eval "$aimnet2_run"

echo -e "\n------- AIMNet2 command was ---------" >> "${arg_gauss[3]}"
echo "?> $aimnet2_run" >> "${arg_gauss[3]}"
echo "---------------------------------" >> "${arg_gauss[3]}"

# Currently, non-singlet spins are not supported explicitly by the interface.
if [[ $multip -ne 1 ]]; then
    echo "WARNING: Gaussian multiplicity S=$multip is not singlet." >> "${arg_gauss[3]}"
    echo "         This is not not explicitly supported. Results are likely wrong without" >> "${arg_gauss[3]}"
    echo "         an appropriate spin argument to AIMNet2!" >> "${arg_gauss[3]}"
fi

# Close log and flush
echo "             Control returned to Gaussian." >> "${arg_gauss[3]}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> "${arg_gauss[3]}"