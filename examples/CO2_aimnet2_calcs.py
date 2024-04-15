import torch
# import numpy as np
from ase import Atoms
# from ase.units import Hartree
# from ase.vibrations import Vibrations
# from aimnet2_ase_opt import optimize, get_dipole_moment, get_charges
from aimnet2ase import AIMNet2Calculator

# Load your trained model
model_path = '/home/passos/AIMNet2/models/aimnet2_wb97m-d3_ens.jpt'

model = torch.jit.load(model_path)

# Create the calculator
calc = AIMNet2Calculator(model)

# Define the molecule
# mol_CO2 = Atoms('CO2', positions=[[-0.0220,  0.0000,  0.0000], 
#                                [ 1.1621,  0.0000,  0.0000],
#                                [-1.1841,  0.0000,  0.0000]])

# mol_CO2 = Atoms('CO2', positions=[[-0.0220,  0.0000,  0.0000], 
#                                [ 1.1621,  0.0000,  0.0000],
#                                [-1.1841,  0.0000,  0.0000]])

from ase.io import read

# Load molecule from .xyz file
mol_CO2 = read('CO.xyz')


# Set the calculator
mol_CO2.set_calculator(calc)

# Optimize the geometry of the molecule
# optimize(mol_CO2, prec=1e-3, steps=1000)

# Calculate energy, forces, charges, dipole moment
energy_CO2 = mol_CO2.get_potential_energy()
forces_CO2 = mol_CO2.get_forces()
frequencies_CO2 = calc.get_frequencies(mol_CO2)
dipole_CO2 = calc.get_dipole_moment(mol_CO2)
vib_CO2 = calc.get_vibrational_modes(mol_CO2)

# Use imported functions for charges, dipole moment, hessian, vibrational frequencies, and vibrational modes
# charges_CO2 = get_charges(mol_CO2)
# dipole_CO2 = get_dipole_moment(mol_CO2)
# hessian_CO2 = get_hessian(mol_CO2)
# vib_CO2 = get_vibrational_modes(mol_CO2)
# freq_CO2 = get_frequencies(mol_CO2)

# charges_CO2 = calc.results['charges']
# dipole_CO2 = mol_CO2.get_dipole_moment()

# dipole_CO2 = calc.results['dipole']

# Calculate the magnitude of the dipole moment vector
# dipole_magnitude_CO2 = np.linalg.norm(dipole_CO2)

# # Calculate Hessian matrix for CO2
# hessian_CO2 = mol_CO2.calc.get_hessian(mol_CO2)

# # Get the indices of all atoms in mol_CO2
# indices_CO2 = list(range(len(mol_CO2)))

# vib_CO2 = Vibrations(mol_CO2, indices_CO2)
# vib_CO2.run()  # Run the vibrational analysis
# freq_CO2 = vib_CO2.get_frequencies()

# print(f'Predicted energy for CO2: {energy_CO2:.8f} eV')
# print(f'Forces on CO2:\n{forces_CO2}')  
# print(f'Charges on CO2 atoms:\n{charges_CO2}')
# # print(f'Dipole moment of CO2: {dipole_magnitude_CO2:.4f} D')
# print(f'Dipole moment of CO2: {dipole_CO2:.4f} D')
# print(f'Vibrational frequencies of CO2 (cm^-1):\n{freq_CO2}')

# energy_CO2_hartree = energy_CO2 / Hartree
# print(f'Predicted energy for CO2: {energy_CO2_hartree:.8f} Hartree')


print(calc.results)