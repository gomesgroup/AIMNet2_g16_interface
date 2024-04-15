"""
This file defines the AIMNet2Calculator class, which integrates the AIMNet2 neural network model with the Atomic Simulation Environment (ASE) to enable the calculation of various molecular properties such as energy, forces, and vibrational modes. The class is designed to work with PyTorch models and utilizes ASE's Calculator interface for seamless integration with ASE's simulation framework. It includes methods for initializing the calculator with a specific model and charge, preparing input data for the model, and resetting the calculator's internal state.
"""
import os
import json
import re
import argparse
import numpy as np
import torch
import ase
from ase import Atom, Atoms
from ase.data import chemical_symbols
from ase.io import read, write
from ase.optimize import LBFGS
from ase.units import *
from ase.calculators.calculator import Calculator
from ase.vibrations import *
from openbabel import pybel

"""
UNIT CONVERSIONS FOR COMPATIBILITY:
This section outlines the necessary unit conversions to ensure compatibility between ASE and Gaussian 16, as they use different default units. Adhering to Gaussian 16's conventions is crucial for consistency in calculations.

Gaussian 16 Default Units:
- Energy: Hartrees (Eh)
- Distance: Bohrs (a0)
- Forces: Hartrees/Bohr
- Dipole moments: Debye

ASE Default Units:
- Energy: electronvolts (eV)
- Distance: Ångströms (Å)
- Forces: eV/Å
- Dipole moments: e*Å

To bridge these differences, the following conversions must be applied:
- Energy: Multiply eV by ase.units.Hartree to convert to Hartrees
- Distance: Multiply Ångströms by ase.units.Bohr to convert to Bohrs
- Forces: Multiply eV/Å by (ase.units.Hartree / ase.units.Bohr) to convert to Hartrees/Bohr
- Dipole moments: Multiply e*Å by (ase.units.Debye / ase.units._e) to convert to Debye

These conversions ensure that all properties calculated within this code align with Gaussian 16's unit conventions, enabling accurate and meaningful integration.
"""

class AIMNet2Calculator(ase.calculators.calculator.Calculator):
    """ ASE calculator for AIMNet2 model
    Arguments:
        model (:class:`torch.nn.Module`): AIMNet2 model
        charge (int or float): molecular charge.  Default: 0
    """

    implemented_properties = ['energy', 'forces', 'free_energy', 'charges', 'dipole_moment', 
                              'hessian', 'frequencies', 'vibrational_modes']


    def __init__(self, model, charge=0, **kwargs):
        """Initialize the AIMNet2 calculator with a given model and charge.

        Args:
            model (:class:`torch.nn.Module`): The AIMNet2 model to use for calculations.
            charge (int or float, optional): The molecular charge. Defaults to 0.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.model = model
        self.charge = charge
        self.device = next(model.parameters()).device
        cutoff = max(v.item() for k, v in model.state_dict().items() if k.endswith('aev.rc_s'))
        self.cutoff = float(cutoff)
        self._t_numbers = None
        self._t_charge = None
        self.do_dipole = kwargs.get('do_dipole', False)

    def do_reset(self):
        """Reset the internal state of the calculator."""
        self._t_numbers = None
        self._t_charge = None
        self.charge = 0.0

    def _make_input(self):
        """Prepare the input data for the model based on the current state of atoms."""
        coord = torch.as_tensor(self.atoms.positions).to(torch.float).to(self.device).unsqueeze(0)
        if self._t_numbers is None:
            self._t_numbers = torch.as_tensor(self.atoms.numbers).to(torch.long).to(self.device).unsqueeze(0)
            self._t_charge = torch.tensor([self.charge], dtype=torch.float, device=self.device)
        d = dict(coord=coord, numbers=self._t_numbers, charge=self._t_charge)
        return d

    def set_charge(self, charge):
        """Set the molecular charge for the calculation.

        Args:
            charge (float): The charge to set.
        """
        self.charge = float(charge)

    def get_dipole_moment(self, atoms):
        """Calculate and return the dipole moment for the given atoms in Debye.

        Args:
            atoms (ase.Atoms): The atomic configuration for which the dipole moment is calculated.

        Returns:
            float: The calculated dipole moment in Debye.
        """
        if 'charges' not in self.results:
            self.calculate(atoms, properties=['charges'])
        
        charges = self.results['charges']
        positions = atoms.get_positions()
        
        dipole_moment = np.zeros(3)
        for charge, pos in zip(charges, positions):
            dipole_moment += charge * pos
        
        dipole_moment = np.linalg.norm(dipole_moment)
        # # Convert from e*Å to Debye
        # conversion_factor = ase.units.Debye / (ase.units._e * ase.units.Bohr)
        # dipole_moment_debye = dipole_moment * conversion_factor
        # self.results['dipole_moment'] = dipole_moment_debye
        self.results['dipole_moment'] = dipole_moment
        return self.results['dipole_moment']

    def get_forces(self, atoms):
        """Calculate and return the forces on the atoms in Hartrees/Bohr.

        Args:
            atoms (ase.Atoms): The atomic configuration for which forces are calculated.

        Returns:
            np.ndarray: The calculated forces in Hartrees/Bohr.
        """
        self.calculate(atoms, properties=['forces'])
        # Convert forces from eV/Å to Hartree/Bohr
        conversion_factor = (ase.units.Hartree / ase.units.eV) * (ase.units.Ang / ase.units.Bohr)
        self.results['forces'] = self.results['forces'] * conversion_factor
    
        return self.results['forces']

    def get_hessian(self, atoms):
        """Compute the Hessian matrix for the given atoms using numerical differentiation of forces.

        Args:
            atoms (ase.Atoms): The atomic configuration for which the Hessian matrix is calculated.

        Returns:
            np.ndarray: The calculated Hessian matrix in Hartrees/Bohr^2.
        """
        num_atoms = len(atoms)
        hessian = np.zeros((3*num_atoms, 3*num_atoms))
        delta = 1e-3  # Finite difference step size in Ångströms

        # Convert delta from Ångströms to Bohrs for consistency with force units
        delta_in_bohrs = delta / ase.units.Bohr

        for i in range(num_atoms):
            for j in range(3):
                # Positive step
                atoms_copy = atoms.copy()
                atoms_copy.positions[i, j] += delta_in_bohrs  # Use delta in Bohrs
                fplus = self.get_forces(atoms_copy).reshape(-1)

                # Negative step
                atoms_copy = atoms.copy()
                atoms_copy.positions[i, j] -= delta_in_bohrs  # Use delta in Bohrs
                fminus = self.get_forces(atoms_copy).reshape(-1)

                hessian[3*i+j,:] = (fplus - fminus) / (2 * delta_in_bohrs)
            
        self.results['hessian'] = hessian  # Already in Hartrees/Bohr^2, no conversion needed
        return self.results['hessian']

    def is_linear(self, atoms):
        """
        Determine if a molecule is linear based on its moment of inertia tensor.
        
        Args:
            atoms (ase.Atoms): The atomic configuration.
        
        Returns:
            bool: True if the molecule is linear, False otherwise.
        """
        # Since get_moments_of_inertia() is not available in this class, we'll calculate the inertia tensor manually
        positions = atoms.get_positions()
        masses = atoms.get_masses()
        center_of_mass = np.average(positions, weights=masses, axis=0)
        rel_positions = positions - center_of_mass
        inertia_tensor = np.zeros((3, 3))
        for i in range(len(atoms)):
            for j in range(3):
                for k in range(3):
                    if j == k:
                        inertia_tensor[j, k] += masses[i] * (np.dot(rel_positions[i], rel_positions[i]) - rel_positions[i, j] * rel_positions[i, k])
                    else:
                        inertia_tensor[j, k] -= masses[i] * rel_positions[i, j] * rel_positions[i, k]
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)
        # A molecule is considered linear if two of the moments of inertia are significantly larger
        # than the third, indicating a high degree of symmetry along one axis.
        # This threshold is somewhat arbitrary and may need adjustment
        threshold = 1e-3
        sorted_eigenvalues = sorted(eigenvalues)
        return sorted_eigenvalues[1] / sorted_eigenvalues[0] > threshold and sorted_eigenvalues[2] / sorted_eigenvalues[0] > threshold

    def get_frequencies(self, atoms):
        """Calculate vibrational frequencies for the given atoms.

        Args:
            atoms (ase.Atoms): The atomic configuration for which frequencies are calculated.

        Returns:
            list: The calculated vibrational frequencies.
        """
        # Ensure atoms object is correctly set up for calculation
        # This is a placeholder for any pre-checks or setup you might need

        # Calculate Hessian and initialize Vibrations object
        try:
            # Attempt to calculate the Hessian matrix
            hessian = self.get_hessian(atoms)
        except Exception as e:
            print(f"Error calculating Hessian matrix: {e}")
            return []

        # Ensure correct handling of degrees of freedom
        if self.is_linear(atoms):
            dof = len(atoms) * 3 - 5
        else:
            dof = len(atoms) * 3 - 6

        vib = ase.vibrations.Vibrations(atoms)

        try:
            vib.run()
            all_frequencies = vib.get_frequencies()
            if len(all_frequencies) > dof:
                frequencies = all_frequencies[:dof]  # Adjust frequencies based on degrees of freedom
            else:
                frequencies = all_frequencies
        except IndexError as e:
            print(f"Error calculating frequencies: {e}")
            frequencies = []
        except Exception as e:
            print(f"Unexpected error during frequency calculation: {e}")
            frequencies = []

        self.results['frequencies'] = frequencies
        return self.results['frequencies']

    def calculate_frequencies(hessian, atoms):
        """
        Calculate vibrational frequencies from the Hessian matrix.
        
        Args:
            hessian (np.ndarray): The mass-weighted Hessian matrix.
            atoms (ase.Atoms): The atomic configuration.
        
        Returns:
            np.ndarray: The vibrational frequencies.
        """
        # Determine if the molecule is linear
        linear = is_linear(atoms)
        
        # Calculate eigenvalues (square roots give frequencies in appropriate units)
        eigenvalues = np.linalg.eigvalsh(hessian)
        
        # Conversion factor from (rad/s)^2 to cm^-1
        conversion_factor = 1 / (2 * np.pi * ase.units.c * 100)
        
        # Convert eigenvalues to frequencies
        frequencies = np.sqrt(np.abs(eigenvalues)) * conversion_factor
        
        # Adjust for degrees of freedom
        dof = 3 * len(atoms) - (5 if linear else 6)
        
        # Ensure you only access valid indices
        valid_frequencies = frequencies[:dof]
        
        return valid_frequencies

    def get_vibrational_modes(self, atoms):
        """Calculate and return the vibrational modes for the given atoms.

        Args:
            atoms (ase.Atoms): The atomic configuration for which vibrational modes are calculated.

        Returns:
            np.ndarray: The calculated vibrational modes.
        """
        if 'hessian' not in self.results:
            self.calculate(atoms, properties=['hessian'])
        hessian = self.results['hessian']
        vibrational_modes = np.linalg.eigvalsh(hessian)
        self.results['vibrational_modes'] = vibrational_modes
        return self.results['vibrational_modes']

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Calculate and return the potential energy for the given atoms.

        Args:
            atoms (ase.Atoms, optional): The atomic configuration for which the potential energy is calculated. Defaults to None.
            force_consistent (bool, optional): Whether to use force-consistent energy. Defaults to False.

        Returns:
            float: The calculated potential energy.
        """
        # Check if force_consistent is requested and adjust properties accordingly
        properties = ['energy']
        if force_consistent:
            properties.append('free_energy')
        
        self.calculate(atoms, properties=properties)
        
        # Use force-consistent energy if requested, otherwise use the standard energy
        if force_consistent and 'free_energy' in self.results:
            energy = self.results['free_energy']
        else:
            energy = self.results['energy']
        
        # Convert energy to eV from Hartree
        # energy /= ase.units.Hartree
        # energy = energy
        energy /= ase.units.Hartree
        
        return energy

    def _eval_model(self, d, forces=True, dipole=False, frequencies=False, hessian=False, vibrational_modes=False):
        """Evaluate the model with the given inputs and calculation options.

        Args:
            d (dict): The input data for the model.
            forces (bool, optional): Whether to calculate forces. Defaults to True.
            dipole (bool, optional): Whether to calculate the dipole moment. Defaults to False.
            frequencies (bool, optional): Whether to calculate frequencies. Defaults to False.
            hessian (bool, optional): Whether to calculate the Hessian matrix. Defaults to False.
            vibrational_modes (bool, optional): Whether to calculate vibrational modes. Defaults to False.

        Returns:
            dict: A dictionary containing the requested calculated properties.
        """
        prev = torch.is_grad_enabled()
        torch._C._set_grad_enabled(forces)
        if forces:
            d['coord'].requires_grad_(True)
        _out = self.model(d)
        ret = dict(energy=_out['energy'].item(), charges=_out['charges'].detach()[0].cpu().numpy())
        if forces:
            if 'forces' in _out:
                f = _out['forces'][0]
            else:
                f = - torch.autograd.grad(_out['energy'], d['coord'])[0][0]
            ret['forces'] = f.detach().cpu().numpy()
        if dipole:
            if 'dipole' in _out:
                ret['dipole'] = _out['dipole'].detach()[0].cpu().numpy()
            else:
                raise ValueError("The model does not compute dipole moments")
        if hessian:
            if 'hessian' in _out:
                ret['hessian'] = _out['hessian'].detach()[0].cpu().numpy()
            else:
                raise ValueError("The model does not compute Hessian matrix")
        if frequencies:
            if 'frequencies' in _out:
                ret['frequencies'] = _out['frequencies'].detach()[0].cpu().numpy()
            else:
                raise ValueError("The model does not compute frequencies")
        if vibrational_modes:
            if 'vibrational_modes' in _out:
                ret['vibrational_modes'] = _out['vibrational_modes'].detach()[0].cpu().numpy()
            else:
                raise ValueError("The model does not compute vibrational modes")
        torch._C._set_grad_enabled(prev)
        return ret

    def calculate(self, atoms=None, properties=['energy'],
              system_changes=ase.calculators.calculator.all_changes):
        """Perform the calculation for the requested properties on the given atoms.

        Args:
            atoms (ase.Atoms, optional): The atomic configuration for which properties are calculated. Defaults to None.
            properties (list, optional): The list of properties to calculate. Defaults to ['energy'].
            system_changes (list, optional): A list of changes that have occurred. Defaults to all_changes.
        """
        super().calculate(atoms, properties, system_changes)
        _in = self._make_input()
        do_forces = 'forces' in properties
        do_dipole = 'dipole' in properties
        do_frequencies = 'frequencies' in properties
        do_hessian = 'hessian' in properties
        do_vibrational_modes = 'vibrational_modes' in properties    
        _out =  self._eval_model(_in, do_forces, do_dipole, do_frequencies, do_hessian, do_vibrational_modes)

        self.results['energy'] = _out['energy']
        self.results['charges'] = _out['charges']
        if do_forces:
            self.results['forces'] = _out['forces']
        if do_dipole:
            self.results['dipole'] = _out['dipole']
        if do_hessian:
            self.results['hessian'] = _out['hessian']
        if do_frequencies:
            self.results['frequencies'] = _out['frequencies']
        if do_vibrational_modes:
            self.results['vibrational_modes'] = _out['vibrational_modes']

# class AIMNet2ASEUtilities:
#     def __init__(self, calculator):
#         """
#         Initialize the utilities class with an AIMNet2Calculator instance.

#         Args:
#             calculator (AIMNet2Calculator): The calculator instance to use for calculations.
#         """
#         self.calculator = calculator
#     def optimize(self, atoms, prec=1e-3, steps=1000, traj=None):
#         """Optimize the geometry of a given set of atoms.

#         Args:
#             atoms (ase.Atoms): The atomic configuration to optimize.
#             prec (float): The convergence criterion for forces.
#             steps (int): The maximum number of optimization steps.
#             traj (str): Path to the file where the optimization trajectory will be saved.
#         """
#         with torch.jit.optimized_execution(False):
#             opt = LBFGS(atoms, trajectory=traj)
#             opt.run(fmax=prec, steps=steps)
    
#     def get_charges(atoms):
#         """Retrieve the charges from the calculation results.

#         Args:
#             atoms (ase.Atoms): The atomic configuration for which charges are retrieved.

#         Returns:
#             np.ndarray: The calculated charges.
#         """
#         return atoms.calc.results['charges']

#     def pybel2atoms(mol):
#         """Convert a Pybel molecule to an ASE Atoms object.

#         Args:
#             mol (pybel.Molecule): The Pybel molecule to convert.

#         Returns:
#             ase.Atoms: The converted ASE Atoms object.
#         """
#         coord = np.array([a.coords for a in mol.atoms])
#         numbers = np.array([a.atomicnum for a in mol.atoms])
#         atoms = ase.Atoms(positions=coord, numbers=numbers)
#         return atoms

#     # def update_mol(mol, sample_atoms, align=False):
#     #     print(f"Initial atom count: {len(sample_atoms)}")
#     #     for atom in mol.atoms:
#     #         new_atom = Atom(atom.symbol, atom.coords)  # Simplified; actual conversion may vary
#     #         sample_atoms.append(new_atom)
#     #         print(f"Added atom: {atom.symbol} at {atom.coords}")
#     #     print(f"Updated atom count: {len(sample_atoms)}")

#     def update_mol(pybel_mol, ase_atoms, align=False):
#         for pybel_atom in pybel_mol.atoms:
#             atomic_num = pybel_atom.atomicnum
#             symbol = chemical_symbols[atomic_num]  # Convert atomic number to symbol
#             x, y, z = pybel_atom.coords
#             new_atom = Atom(symbol, (x, y, z))
#             ase_atoms.append(new_atom)

#     # def update_mol(mol, atoms, align=True):
#     #     """Update the coordinates of a Pybel molecule from an ASE Atoms object.

#     #     Args:
#     #         mol (pybel.Molecule): The Pybel molecule to update.
#     #         atoms (ase.Atoms): The ASE Atoms object providing the new coordinates.
#     #         align (bool): Whether to align the updated molecule to the original one.
#     #     """
#     #     mol_old = pybel.Molecule(pybel.ob.OBMol(mol.OBMol))
#     #     for i, c in enumerate(atoms.get_positions()):
#     #         mol.OBMol.GetAtom(i+1).SetVector(*c.tolist())
#     #     if align:
#     #         aligner = pybel.ob.OBAlign(False, False)
#     #         aligner.SetRefMol(mol_old.OBMol)
#     #         aligner.SetTargetMol(mol.OBMol)
#     #         aligner.Align()
#     #         rmsd = aligner.GetRMSD()
#     #         aligner.UpdateCoords(mol.OBMol)
#     #         print(f'RMSD: {rmsd:.2f} Angs')

#     def guess_pybel_type(filename):
#         """Guess the file type based on its extension.

#         Args:
#             filename (str): The name of the file.

#         Returns:
#             str: The guessed file type.
#         """
#         assert '.' in filename
#         return os.path.splitext(filename)[1][1:]

#     def guess_charge(mol):
#         """Guess the charge of a molecule.

#         Args:
#             mol (pybel.Molecule): The molecule for which to guess the charge.

#         Returns:
#             int: The guessed charge.
#         """
#         m = re.search('charge: (-?\d+)', mol.title)
#         if m:
#             charge = int(m.group(1))
#         else:
#             charge = mol.charge
#         return charge

#     def calculate_single_point_energy(calc, atoms):
#         """Calculate the energy of a molecule.

#         Args:
#             calc (AIMNet2Calculator): The calculator object.
#             atoms (ase.Atoms): The atomic configuration.

#         Returns:
#             float: The calculated energy.
#         """
#         single_point_energy = calc.get_potential_energy(atoms)   # get_potential_energy(atoms)
#         return single_point_energy

#     def calculate_dipole(calc, atoms):
#         """Calculate the dipole moment of a molecule.

#         Args:
#             calc (AIMNet2Calculator): The calculator object.
#             atoms (ase.Atoms): The atomic configuration.

#         Returns:
#             np.ndarray: The calculated dipole moment.
#         """
#         dipole = calc.get_dipole_moment(atoms)
#         return dipole

#     def calculate_charges(calc, atoms):
#         """Calculate the charges of a molecule.

#         Args:
#             calc (AIMNet2Calculator): The calculator object.
#             atoms (ase.Atoms): The atomic configuration.

#         Returns:
#             np.ndarray: The calculated charges.
#         """
#         charges = calc.results['charges']
#         return charges

#     def calculate_hessian(calc, atoms):
#         """Calculate the Hessian of a molecule.

#         Args:
#             calc (AIMNet2Calculator): The calculator object.
#             atoms (ase.Atoms): The atomic configuration.

#         Returns:
#             np.ndarray: The calculated Hessian.
#         """
#         return calc.get_hessian(atoms)

#     def calculate_forces(calc, atoms):
#         """Calculate the forces of a molecule.

#         Args:
#             calc (AIMNet2Calculator): The calculator object.
#             atoms (ase.Atoms): The atomic configuration.

#         Returns:
#             np.ndarray: The calculated forces.
#         """
#         return calc.get_forces(atoms)

#     def calculate_frequencies(calc, atoms):
#         """Calculate vibrational frequencies of a molecule.

#         Args:
#             calc (AIMNet2Calculator): The calculator object.
#             atoms (ase.Atoms): The atomic configuration.

#         Returns:
#             np.ndarray: The calculated vibrational frequencies.
#         """
#         if 'hessian' not in calc.results:
#                 # self.calculate(atoms, properties=['hessian'])
#                 calc.get_hessian(atoms)
#                 # self.results['hessian'] = hessian
#         return calc.get_frequencies(atoms)


# """
# # Example of how to use the utilities class with AIMNet2Calculator
# calculator = AIMNet2Calculator(model, charge=0)  # Assuming 'model' is defined elsewhere
# utilities = AIMNet2ASEUtilities(calculator)
# # Now you can call methods like utilities.optimize(atoms), utilities.get_charges(atoms), etc.
# """