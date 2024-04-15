#!/bin/bash

read atoms derivs charge spin < $2

#Create temporary .xyz file
#the element index should be replaced with element name, and the coordinate should be convert to Angstrom

echo "Generating mol.tmp"
cat >> mol.tmp <<EOF
$atoms
$(sed -n 2,$(($atoms+1))p < $2 | cut -c 1-72)
EOF

python ../aimnet2_grad.py > $3  # $3 is just a filename 

if [ "$derivs" -eq 2 ];then
  python aimnet2_hess.py >> $3 
fi

# rm -rf mol.tmp


