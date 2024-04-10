"""Module for optimizing molecular geometries using AIMNet2 and ASE."""
import os
import re
import ase.units
from ase.optimize import LBFGS
import torch
from openbabel import pybel
import numpy as np
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

def update_mol(mol, atoms, align=True):
    """Update the coordinates of a Pybel molecule from an ASE Atoms object.

    Args:
        mol (pybel.Molecule): The Pybel molecule to update.
        atoms (ase.Atoms): The ASE Atoms object providing the new coordinates.
        align (bool): Whether to align the updated molecule to the original one.
    """
    mol_old = pybel.Molecule(pybel.ob.OBMol(mol.OBMol))
    for i, c in enumerate(atoms.get_positions()):
        mol.OBMol.GetAtom(i+1).SetVector(*c.tolist())
    if align:
        aligner = pybel.ob.OBAlign(False, False)
        aligner.SetRefMol(mol_old.OBMol)
        aligner.SetTargetMol(mol.OBMol)
        aligner.Align()
        rmsd = aligner.GetRMSD()
        aligner.UpdateCoords(mol.OBMol)
        print(f'RMSD: {rmsd:.2f} Angs')

def guess_pybel_type(filename):
    """Guess the file type based on its extension.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The guessed file type.
    """
    assert '.' in filename
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

def calculate_properties(calc, atoms):
    """Calculate various properties of a molecule.

    Args:
        calc (AIMNet2Calculator): The calculator object.
        atoms (ase.Atoms): The atomic configuration.

    Returns:
        tuple: A tuple containing energy, forces, dipole, charges, and frequencies.
    """
    energy = calc.get_potential_energy(atoms)
    dipole = calc.get_dipole_moment(atoms)
    charges = calc.results['charges']

    return energy, dipole, charges

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Calculate properties of molecules using AIMNet2 and ASE.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained AIMNet2 model.')
    parser.add_argument('--in_file', type=str, required=True, help='Input file containing molecule(s) in a supported format.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file to write the calculated properties.')
    parser.add_argument('--hessian', type=bool, default=False, help='Whether to calculate the Hessian.')
    parser.add_argument('--forces', type=bool, default=False, help='Whether to calculate the forces.')
    parser.add_argument('--frequencies', type=bool, default=False, help='Whether to calculate vibrational frequencies.')
    parser.add_argument('--charge', type=int, default=None, help='Molecular charge (default: check molecule title for "charge: {int}" or use OpenBabel to guess).')
    parser.add_argument('--traj', type=str, default=None, help='Trajectory file for optimization.')
    parser.add_argument('--fmax', type=float, default=5e-3, help='Force convergence criterion for optimization.')
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(args.model, map_location=device)
    calc = AIMNet2Calculator(model)

    in_format = guess_pybel_type(args.in_file)
    out_format = guess_pybel_type(args.out_file)

    with open(args.out_file, 'w', encoding='utf-8') as f:
        for mol in pybel.readfile(in_format, args.in_file):
            atoms = pybel2atoms(mol)
            charge = args.charge if args.charge is not None else guess_charge(mol)

            calc.do_reset()
            calc.set_charge(charge)
            atoms.set_calculator(calc)

            optimize(atoms, prec=args.fmax, steps=2000, traj=args.traj)
            energy, dipole, charges = calculate_properties(calc, atoms)
            if args.hessian:
                hessian = calculate_hessian(calc, atoms)
                print(f"Hessian: {hessian}")
            if args.forces:
                forces = calculate_forces(calc, atoms)
                print(f"Forces: {forces}")
            if args.frequencies:
                frequencies = calculate_frequencies(calc, atoms)
                print(f"Vibrational Frequencies (cm^-1): {frequencies}")

            print(f"Energy: {energy:.6f} Ha")

            print(f"Dipole Moment: {dipole:.4f} D")
            print(f"Charges: {charges}")
            

            update_mol(mol, atoms, align=False)
            f.write(mol.write(out_format))
            f.flush()

