import unittest
import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from ase.units import Hartree
from aimnet2ase import AIMNet2Calculator
from aimnet2_ase_opt import optimize, get_frequencies, get_dipole_moment, calculate_properties

model_path = '/home/passos/AIMNet2/models/aimnet2_wb97m-d3_ens.jpt'

class TestAIMNet2ASE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = torch.jit.load(model_path)
        cls.calc = AIMNet2Calculator(cls.model)

    def test_molecules_from_xyz(self):
        input_file = 'molecules.xyz'  # This file should exist in the same directory
        molecules = read(input_file, index=':')
        
        results = []
        for mol in molecules:
            mol_name = mol.get_chemical_formula()
            mol.set_calculator(self.calc)
            
            # Perform calculations
            optimize(mol, prec=1e-3, steps=100)
            energy, forces, dipole, charges, frequencies = calculate_properties(mol)
            energy_hartree = energy / Hartree
            
            # Append results
            results.append({
                'name': mol_name,
                'energy': energy_hartree,
                'forces': forces,
                'dipole': dipole,
                'charges': charges,
                'frequencies': frequencies
            })
            
            # Write updated molecule to an output xyz file
            write(f'{mol_name}_out.xyz', mol)
        
        # Write results to an output file
        with open('molecules_results.txt', 'w') as f:
            for result in results:
                f.write(f"Name: {result['name']}\n")
                f.write(f"Energy (Hartree): {result['energy']:.6f}\n")
                f.write(f"Dipole Moment: {result['dipole']:.4f} D\n")
                f.write(f"Vibrational Frequencies (cm^-1): {result['frequencies']}\n")
                f.write("Charges: {}\n".format(", ".join(map(str, result['charges']))))
                f.write("Forces:\n{}\n\n".format(result['forces']))

if __name__ == '__main__':
    unittest.main()

# import unittest
# import numpy as np
# import torch
# from ase import Atoms
# from ase.units import Hartree
# from ase.vibrations import Vibrations
# from aimnet2ase import AIMNet2Calculator
# from aimnet2_ase_opt import optimize, get_frequencies, get_dipole_moment

# model_path = '/home/passos/AIMNet2/models/aimnet2_wb97m-d3_ens.jpt'

# class TestAIMNet2ASE(unittest.TestCase):

#     def setUp(self):
#         model = torch.jit.load(model_path)
#         self.calc = AIMNet2Calculator(model)
        
#         self.mol = Atoms('H2O', positions=[[0,0,0], [0,0,1], [0,1,0]])
#         self.mol.set_calculator(self.calc)

#     def test_single_point(self):
#         energy = self.mol.get_potential_energy() / Hartree
#         self.assertIsInstance(energy, float)
#         self.assertAlmostEqual(energy, -76.33637071653648, places=4)

#     def test_forces(self):
#         forces = self.mol.get_forces()
#         self.assertEqual(forces.shape, (3,3))

#     def test_optimize(self):
#         optimize(self.mol, prec=1e-3, steps=100)
#         energy = self.mol.get_potential_energy() / Hartree
#         self.assertLess(energy, -76.3)

#     def test_charges(self):
#         self.mol.get_potential_energy()  # Ensure the calculation is performed
#         charges = self.calc.results['charges']
#         self.assertEqual(len(charges), len(self.mol))

#     def test_dipole_moment(self):
#         dipole = self.mol.get_dipole_moment()
#         dipole_magnitude = np.linalg.norm(dipole)
#         self.assertIsInstance(dipole_magnitude, float)

#     def test_hessian(self):
#         hessian = self.mol.calc.get_hessian(self.mol)
#         self.assertEqual(hessian.shape, (3*len(self.mol), 3*len(self.mol)))

#     def test_frequencies(self):
#         freqs = get_frequencies(self.mol) 
#         geometry = 'linear' if np.all(self.mol.get_angular_momentum() == 0) else 'non-linear'
#         expected_modes = 3*len(self.mol) - (5 if geometry == 'linear' else 6)
        
#         indices = list(range(len(self.mol)))
#         vib = Vibrations(self.mol, indices=indices)
#         vib.run()
#         freq = vib.get_frequencies()
#         print(f"Geometry: {geometry}, Number of frequencies: {len(freqs)}, Expected vibrational modes: {expected_modes}")
        
#         for i, frequency in enumerate(freqs, start=1):
#             print(f"Vibrational mode {i}: {frequency:.2f} cm^-1")
        
#         self.assertTrue(all(freqs > 0))

        

#     def test_another_molecule(self):
#         mol = Atoms('CO2', positions=[[-0.0220,  0.0000,  0.0000], 
#                                     [ 1.1621,  0.0000,  0.0000],
#                                     [-1.1841,  0.0000,  0.0000]])
#         mol.set_calculator(self.calc)
#         energy = mol.get_potential_energy() / Hartree
#         self.assertAlmostEqual(energy, -188.71663826626724, places=4)

# if __name__ == '__main__':
#     unittest.main()