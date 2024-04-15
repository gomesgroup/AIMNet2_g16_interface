import unittest
from unittest.mock import patch, MagicMock
import aimnet2ase_exec

class TestAimnet2AseExec(unittest.TestCase):

    @patch('aimnet2ase_exec.pybel2atoms')
    def test_optimize(self, mock_pybel2atoms):
        # Mock inputs and expected outputs
        mock_pybel2atoms.return_value = ([], [])
        mock_mol = MagicMock()
        mock_calculator = MagicMock()

        # Call the function under test
        aimnet2ase_exec.optimize(mock_mol, mock_calculator)

        # Assert expected behavior
        mock_pybel2atoms.assert_called_once_with(mock_mol)
        mock_calculator.calculate.assert_called_once()

    @patch('aimnet2ase_exec.guess_charge')
    def test_get_charges(self, mock_guess_charge):
        # Mock inputs and expected outputs
        mock_guess_charge.return_value = [0, 1, -1]
        mock_atoms = [MagicMock(), MagicMock(), MagicMock()]

        # Call the function under test
        charges = aimnet2ase_exec.get_charges(mock_atoms)

        # Assert expected behavior
        mock_guess_charge.assert_called_once_with(mock_atoms)
        self.assertEqual(charges, [0, 1, -1])

    @patch('aimnet2ase_exec.guess_pybel_type')
    def test_pybel2atoms(self, mock_guess_pybel_type):
        # Mock inputs and expected outputs
        mock_guess_pybel_type.side_effect = ['C', 'H', 'O']
        mock_mol = MagicMock()
        mock_mol.atoms = [MagicMock(), MagicMock(), MagicMock()]

        # Call the function under test
        atoms, charges = aimnet2ase_exec.pybel2atoms(mock_mol)

        # Assert expected behavior
        self.assertEqual(len(atoms), 3)
        self.assertEqual(len(charges), 3)
        mock_guess_pybel_type.assert_has_calls([
            mock.call(mock_mol.atoms[0]),
            mock.call(mock_mol.atoms[1]),
            mock.call(mock_mol.atoms[2])
        ])

    @patch('aimnet2ase_exec.get_charges')
    @patch('aimnet2ase_exec.pybel2atoms')
    def test_update_mol(self, mock_pybel2atoms, mock_get_charges):
        # Mock inputs and expected outputs
        mock_pybel2atoms.return_value = (['C', 'H', 'O'], [0, 1, -1])
        mock_get_charges.return_value = [0, 1, -1]
        mock_mol = MagicMock()
        mock_atoms = [MagicMock(), MagicMock(), MagicMock()]

        # Call the function under test
        aimnet2ase_exec.update_mol(mock_mol, mock_atoms)

        # Assert expected behavior
        mock_pybel2atoms.assert_called_once_with(mock_mol)
        mock_get_charges.assert_called_once_with(mock_atoms)
        for atom, charge in zip(mock_atoms, [0, 1, -1]):
            atom.set_charge.assert_called_once_with(charge)

    def test_guess_pybel_type(self):
        # Mock inputs and expected outputs
        mock_atom = MagicMock()
        mock_atom.type = 'C'

        # Call the function under test
        atom_type = aimnet2ase_exec.guess_pybel_type(mock_atom)

        # Assert expected behavior
        self.assertEqual(atom_type, 'C')

    @patch('aimnet2ase_exec.guess_pybel_type')
    def test_guess_charge(self, mock_guess_pybel_type):
        # Mock inputs and expected outputs
        mock_guess_pybel_type.side_effect = ['C', 'O', 'N']
        mock_atoms = [MagicMock(), MagicMock(), MagicMock()]

        # Call the function under test
        charges = aimnet2ase_exec.guess_charge(mock_atoms)

        # Assert expected behavior
        self.assertEqual(charges, [0, -2, 0])
        mock_guess_pybel_type.assert_has_calls([
            mock.call(mock_atoms[0]),
            mock.call(mock_atoms[1]),
            mock.call(mock_atoms[2])
        ])

    @patch('aimnet2ase_exec.calculate_properties')
    def test_calculate_hessian(self, mock_calculate_properties):
        # Mock inputs and expected outputs
        mock_calculate_properties.return_value = {'hessian': MagicMock()}
        mock_atoms = MagicMock()
        mock_calculator = MagicMock()

        # Call the function under test
        hessian = aimnet2ase_exec.calculate_hessian(mock_atoms, mock_calculator)

        # Assert expected behavior
        mock_calculate_properties.assert_called_once_with(mock_atoms, mock_calculator, ['hessian'])
        self.assertEqual(hessian, mock_calculate_properties.return_value['hessian'])

    @patch('aimnet2ase_exec.calculate_properties')
    def test_calculate_energy(self, mock_calculate_properties):
        # Mock inputs and expected outputs
        mock_calculate_properties.return_value = {'energy': MagicMock()}
        mock_atoms = MagicMock()
        mock_calculator = MagicMock()

        # Call the function under test
        energy = aimnet2ase_exec.calculate_energy(mock_atoms, mock_calculator)

        # Assert expected behavior
        mock_calculate_properties.assert_called_once_with(mock_atoms, mock_calculator, ['energy'])
        self.assertEqual(energy, mock_calculate_properties.return_value['energy'])

if __name__ == '__main__':
    unittest.main()