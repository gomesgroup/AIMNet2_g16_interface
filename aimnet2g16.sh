#!/bin/bash

# This script is used to convert the output of aimnet to the format of Gaussian16

# add the following path `models/aimnet2_wb97m-d3_ens.jpt` 



read atoms derivs charge spin < $2

#Create temporary .xyz file
#the element index should be replaced with element name, and the coordinate should be convert to Angstrom

echo "Generating mol.tmp"
cat >> mol.tmp <<EOF
$atoms
$(sed -n 2,$(($atoms+1))p < $2 | cut -c 1-72)
EOF

# Run aimnet2ase_exec.py
echo "Running aimnet2ase_exec.py"
# the keywords in the g16 job will indicate which arguments to pass to aimnet2ase_exec.py:
# if the g16 job contains the keyword "hess", then aimnet2ase_exec.py should be run with the argument "--hess";
# if the g16 job contains the keyword "calcfc" or "calcall", then aimnet2ase_exec.py should be run with the argument "--forces"
# energy, charges, hessians, and forces are written to the file specified by the third argument
# aimnet

python calculators/ase/aimnet2ase_exec.py > $3  # $3 is just a filename 

# if [ "$derivs" -eq 2 ];then
#   python aimnet2_hess.py >> $3 
# fi

rm -rf mol.tmp

