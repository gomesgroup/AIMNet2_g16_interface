"""Module with functions for using AIMNet2 from ASE."""
import os
import json
import re
import ase.units
from ase.optimize import LBFGS
import torch
from openbabel import pybel
import numpy as np
import argparse
from ase.io import read, write
from ase import Atom, Atoms
from ase.data import chemical_symbols
from aimnet2ase import AIMNet2Calculator

def optimize(atoms, prec=1e-3, steps=1000, traj=None):
    """Optimize the geometry of a given set of atoms.

    Args:
        atoms (ase.Atoms): The atomic configuration to optimize.
        prec (float): The convergence criterion for forces.
        steps (int): The maximum number of optimization steps.
        traj (str): Path to the file where the optimization trajectory will be saved.
    """
    with torch.jit.optimized_execution(False):
        opt = LBFGS(atoms, trajectory=traj)
        opt.run(prec, steps)

def get_charges(atoms):
    """Retrieve the charges from the calculation results.

    Args:
        atoms (ase.Atoms): The atomic configuration for which charges are retrieved.

    Returns:
        np.ndarray: The calculated charges.
    """
    return atoms.calc.results['charges']

def pybel2atoms(mol):
    """Convert a Pybel molecule to an ASE Atoms object.

    Args:
        mol (pybel.Molecule): The Pybel molecule to convert.

    Returns:
        ase.Atoms: The converted ASE Atoms object.
    """
    coord = np.array([a.coords for a in mol.atoms])
    numbers = np.array([a.atomicnum for a in mol.atoms])
    atoms = ase.Atoms(positions=coord, numbers=numbers)
    return atoms

# def update_mol(mol, sample_atoms, align=False):
#     print(f"Initial atom count: {len(sample_atoms)}")
#     for atom in mol.atoms:
#         new_atom = Atom(atom.symbol, atom.coords)  # Simplified; actual conversion may vary
#         sample_atoms.append(new_atom)
#         print(f"Added atom: {atom.symbol} at {atom.coords}")
#     print(f"Updated atom count: {len(sample_atoms)}")

def update_mol(pybel_mol, ase_atoms, align=False):
    for pybel_atom in pybel_mol.atoms:
        atomic_num = pybel_atom.atomicnum
        symbol = chemical_symbols[atomic_num]  # Convert atomic number to symbol
        x, y, z = pybel_atom.coords
        new_atom = Atom(symbol, (x, y, z))
        ase_atoms.append(new_atom)

# def update_mol(mol, atoms, align=True):
#     """Update the coordinates of a Pybel molecule from an ASE Atoms object.

#     Args:
#         mol (pybel.Molecule): The Pybel molecule to update.
#         atoms (ase.Atoms): The ASE Atoms object providing the new coordinates.
#         align (bool): Whether to align the updated molecule to the original one.
#     """
#     mol_old = pybel.Molecule(pybel.ob.OBMol(mol.OBMol))
#     for i, c in enumerate(atoms.get_positions()):
#         mol.OBMol.GetAtom(i+1).SetVector(*c.tolist())
#     if align:
#         aligner = pybel.ob.OBAlign(False, False)
#         aligner.SetRefMol(mol_old.OBMol)
#         aligner.SetTargetMol(mol.OBMol)
#         aligner.Align()
#         rmsd = aligner.GetRMSD()
#         aligner.UpdateCoords(mol.OBMol)
#         print(f'RMSD: {rmsd:.2f} Angs')

def ein_to_xyz(ein_eou_file, xyz_file):
    """
    Converts an EIn file to an XYZ file.

    Args:
        ein_eou_file (str): Path to the input EIn or EOu file.
        xyz_file (str): Path to the output XYZ file.
    """
    with open(ein_eou_file, 'r') as f_in, open(xyz_file, 'w') as f_out:
        lines = f_in.readlines()
        num_atoms = int(lines[0].split()[0])
        
        f_out.write(f"{num_atoms}\n")
        f_out.write("Converted from EIn or EOu file\n")
        
        for line in lines[1:num_atoms+1]:
            f_out.write(line)

def guess_pybel_type(filename):
    """Guess the file type based on its extension.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The guessed file type.
    """
    
    assert '.' in filename
    extension = os.path.splitext(filename)[1][1:]
    # if extension == 'xyz':
    #     return 'xyz'
    if extension == 'EIn':
        ein_to_xyz(filename, filename+'EIn.xyz')
        return 'xyz'
    elif extension == 'EOu':
        if os.path.exists(filename):
            ein_to_xyz(filename, filename+'EOu.xyz')
        return 'xyz'
        # return 'xyz'
    # else:
    #     raise ValueError(f"Unsupported file type: {extension}")
    return os.path.splitext(filename)[1][1:]

def guess_charge(mol):
    """Guess the charge of a molecule.

    Args:
        mol (pybel.Molecule): The molecule for which to guess the charge.

    Returns:
        int: The guessed charge.
    """
    m = re.search('charge: (-?\d+)', mol.title)
    if m:
        charge = int(m.group(1))
    else:
        charge = mol.charge
    return charge

def calculate_single_point_energy(calc, atoms):
    """Calculate the energy of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        float: The calculated energy.
    """
    single_point_energy = calc.get_potential_energy(atoms)   # get_potential_energy(atoms)
    return single_point_energy

def calculate_dipole(calc, atoms):
    """Calculate the dipole moment of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        np.ndarray: The calculated dipole moment.
    """
    dipole = calc.get_dipole_moment(atoms)
    return dipole

def calculate_charges(calc, atoms):
    """Calculate the charges of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        np.ndarray: The calculated charges.
    """
    charges = calc.results['charges']
    return charges

def calculate_hessian(calc, atoms):
    """Calculate the Hessian of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        np.ndarray: The calculated Hessian.
    """
    return calc.get_hessian(atoms)

def calculate_forces(calc, atoms):
    """Calculate the forces of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        np.ndarray: The calculated forces.
    """
    return calc.get_forces(atoms)

def calculate_frequencies(calc, atoms):
    """Calculate vibrational frequencies of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        np.ndarray: The calculated vibrational frequencies.
    """
    if 'hessian' not in calc.results:
            # self.calculate(atoms, properties=['hessian'])
            calc.get_hessian(atoms)
            # self.results['hessian'] = hessian
    return calc.get_frequencies(atoms)



