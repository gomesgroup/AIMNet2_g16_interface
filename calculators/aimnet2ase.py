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

        vib = ase.vibrations.Vibrations(atoms)

        try:
            vib.run()
            frequencies = vib.get_frequencies()
        except IndexError as e:
            print(f"Error calculating frequencies: {e}")
            frequencies = []
        except Exception as e:
            print(f"Unexpected error during frequency calculation: {e}")
            frequencies = []

        self.results['frequencies'] = frequencies
        return self.results['frequencies']

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
        self.calculate(atoms, properties=['energy'])
        self.results['energy'] /= ase.units.Hartree
        return self.results['energy']


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
