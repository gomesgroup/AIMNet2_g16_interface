import unittest
import os
import sys
import numpy as np
import ase
from ase.build import molecule

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aimnet2ase_exec_fixed import AIMNet2Exec
from aimnet2ase_fixed import AIMNet2Calculator

class TestAIMNet2Exec(unittest.TestCase):
    def setUp(self):
        self.exec = AIMNet2Exec()

    def test_load_model(self):
        # Test loading a valid model
        model_path = "path/to/valid/model.pth"
        self.exec.load_model(model_path)
        self.assertIsNotNone(self.exec.model)

        # Test loading an invalid model
        invalid_model_path = "path/to/invalid/model.pth"
        with self.assertRaises(Exception):
            self.exec.load_model(invalid_model_path)

    def test_get_calculator(self):
        # Test getting a calculator with a valid model
        self.exec.load_model("path/to/valid/model.pth")
        calculator = self.exec.get_calculator(charge=0)
        self.assertIsInstance(calculator, AIMNet2Calculator)

        # Test getting a calculator with an invalid model
        self.exec.model = None
        with self.assertRaises(Exception):
            self.exec.get_calculator(charge=0)

class TestAIMNet2Calculator(unittest.TestCase):
    def setUp(self):
        # Load a dummy model for testing
        self.model = DummyModel()
        self.calculator = AIMNet2Calculator(self.model)

    def test_set_charge(self):
        # Test setting a valid charge
        self.calculator.set_charge(1.0)
        self.assertEqual(self.calculator.charge, 1.0)

        # Test setting an invalid charge
        with self.assertRaises(ValueError):
            self.calculator.set_charge("invalid")

    def test_get_dipole_moment(self):
        # Test getting the dipole moment for a valid molecule
        atoms = molecule("H2O")
        dipole_moment = self.calculator.get_dipole_moment(atoms)
        self.assertIsInstance(dipole_moment, float)

    def test_get_forces(self):
        # Test getting forces for a valid molecule
        atoms = molecule("H2O")
        forces = self.calculator.get_forces(atoms)
        self.assertIsInstance(forces, np.ndarray)
        self.assertEqual(forces.shape, (3, 3))

    def test_get_hessian(self):
        # Test getting the Hessian matrix for a valid molecule
        atoms = molecule("H2O")
        hessian = self.calculator.get_hessian(atoms)
        self.assertIsInstance(hessian, np.ndarray)
        self.assertEqual(hessian.shape, (9, 9))

    def test_get_frequencies(self):
        # Test getting frequencies for a valid molecule
        atoms = molecule("H2O")
        frequencies = self.calculator.get_frequencies(atoms)
        self.assertIsInstance(frequencies, list)

    def test_get_vibrational_modes(self):
        # Test getting vibrational modes for a valid molecule
        atoms = molecule("H2O")
        vibrational_modes = self.calculator.get_vibrational_modes(atoms)
        self.assertIsInstance(vibrational_modes, np.ndarray)

    def test_get_potential_energy(self):
        # Test getting potential energy for a valid molecule
        atoms = molecule("H2O")
        potential_energy = self.calculator.get_potential_energy(atoms)
        self.assertIsInstance(potential_energy, float)

class DummyModel:
    def __init__(self):
        self.state_dict = {
            "aev.rc_s": torch.tensor(5.0)
        }

    def parameters(self):
        return self.state_dict.values()

    def __call__(self, inputs):
        return {
            "energy": torch.tensor(0.0),
            "charges": torch.tensor([0.0, 0.0, 0.0]),
            "forces": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "dipole": torch.tensor([0.0, 0.0, 0.0]),
            "hessian": torch.tensor([[0.0] * 9] * 9),
            "frequencies": torch.tensor([0.0, 0.0, 0.0]),
            "vibrational_modes": torch.tensor([0.0, 0.0, 0.0])
        }

if __name__ == "__main__":
    unittest.main()