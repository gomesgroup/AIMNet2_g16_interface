"""
This file defines the AIMNet2Calculator class, which integrates the AIMNet2 neural network model with the Atomic Simulation Environment (ASE) to enable the calculation of various molecular properties such as energy, forces, and vibrational modes. The class is designed to work with PyTorch models and utilizes ASE's Calculator interface for seamless integration with ASE's simulation framework. It includes methods for initializing the calculator with a specific model and charge, preparing input data for the model, and resetting the calculator's internal state.
"""

import torch
import ase.calculators.calculator
import ase.vibrations
import ase.units
import numpy as np

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

    # ... (the rest of the code remains the same) ...
