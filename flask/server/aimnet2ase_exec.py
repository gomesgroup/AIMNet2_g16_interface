"""Module for executing calculations using AIMNet2 interfaced with ASE."""
import os
import json
import argparse
from ase.io import read, write
from openbabel import pybel
import torch
# from ase import Atom, Atoms
import aimnet2ase
from aimnet2ase import AIMNet2Calculator
from aimnet2ase_functions import *

# if __name__ == '__main__':
def main():    
    parser = argparse.ArgumentParser(description='Calculate properties of molecules using AIMNet2 and ASE.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained AIMNet2 model.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Device to use for calculations (default: cpu). ')
    parser.add_argument('--in_file', type=str, required=True, help='Input file containing molecule(s) in a supported format.')
    parser.add_argument('--out_file', type=str, required=True, help='Output file to write the calculated properties.')
    parser.add_argument('--optimize', action='store_true', help='Whether to perform geometry optimization.')
    parser.add_argument('--traj', action='store_true', help='Whether to save the trajectory file for optimization. The trajectory file name will be automatically generated based on the output file name with "_traj" appended before the extension. Only available if --optimize is used.')
    parser.add_argument('--hessian', action='store_true', help='Whether to calculate the Hessian.')
    parser.add_argument('--forces', action='store_true', help='Whether to calculate the forces.')
    parser.add_argument('--frequencies', action='store_true', help='Whether to calculate vibrational frequencies.')
    parser.add_argument('--charge', type=int, default=None, help='Molecular charge (default: check molecule title for "charge: {int}" or use OpenBabel to guess).')
    parser.add_argument('--fmax', type=float, default=5e-3, help='Force convergence criterion for optimization.')
    parser.add_argument('--out_format', type=str, default='pybel', choices=['pybel', 'json'],
                        help='Output format for the calculated properties (default: pybel).')

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = torch.jit.load(args.model, map_location=device)

    calc = AIMNet2Calculator(model)
    # utils = AIMNet2ASEUtilities(calc)
    
    # Before the loop where you process each molecule, determine the base name for the output file
    base_name, _ = os.path.splitext(args.in_file)
    out_file_base = f"{base_name}_out"
    
    in_format = guess_pybel_type(args.in_file)
    
    # if in_format == 'EIn':
    #     args.in_file = ein_to_xyz(args.in_file, base_name+'.xyz')
    #     in_format = 'xyz'
    out_format = guess_pybel_type(args.out_file)
    # if out_format == 'EOu':
    #     out_format = 'xyz'

    def convert_to_serializable(obj):
        """
        Recursively convert objects in a structure (e.g., dict) that may contain
        non-serializable types (like numpy arrays) into serializable types.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    with open(args.out_file, 'w', encoding='utf-8') as f:
        for mol in pybel.readfile(in_format, args.in_file):
            atoms = pybel2atoms(mol)
            charge = args.charge if args.charge is not None else guess_charge(mol)

            calc.do_reset()
            calc.set_charge(charge)
            # atoms.set_calculator(calc)
            atoms.calc = calc

            single_point_energy = calculate_single_point_energy(calc, atoms)
            dipole = calculate_dipole(calc, atoms)
            charges = calculate_charges(calc, atoms)

            print(f"Electronic Energy (in Ha): {single_point_energy:.8f}")
            print(f"Dipole Moment (in Debye): {dipole} Debye")
            print(f"Charges (in e): {charges}")

            if args.optimize:
                print(f"Energy before geometry optimization: {single_point_energy:.8f} Ha")
                traj_filename = None
                if args.traj:
                    base_name, ext = os.path.splitext(args.out_file)
                    traj_filename = f"{base_name}_traj.traj"
                    print(f"Optimizing geometry for {mol.title} with trajectory output to {traj_filename}.")
                optimize(atoms, prec=args.fmax, steps=2000, traj=traj_filename)
                single_point_energy_opt = calculate_single_point_energy(calc, atoms)
                dipole = calculate_dipole(calc, atoms)
                charges = calculate_charges(calc, atoms)
                if traj_filename:
                    # Convert the .traj file to .xyz format using ASE
                    traj_atoms = read(traj_filename)
                    xyz_filename = traj_filename.replace('.traj', '.xyz')
                    write(xyz_filename, traj_atoms)
                else:
                    print(f"Optimizing geometry for {mol.title}:")
                    optimize(atoms, prec=args.fmax, steps=2000)
                    single_point_energy_opt = calculate_single_point_energy(calc, atoms)
                    dipole = calculate_dipole(calc, atoms)
                    charges = calculate_charges(calc, atoms)
                    
                print(f"Energy after optimization (in Ha): {single_point_energy_opt:.8f}")
                energy_change_Ha = single_point_energy_opt - single_point_energy
                energy_change_kcal = energy_change_Ha * 627.5095
                print(f"Energy change: {energy_change_Ha:.8f} Ha or {energy_change_kcal:.3f} kcal/mol")
                print(f"Dipole Moment after optimization: {dipole}")
                print(f"Charges after optimization: {charges}")
                # print(f"RMSD after optimization: {atoms.get_rmsd():.4f} Angs")
            #     print(f"Max force after optimization: {atoms.get_forces().max():.4f} Ha/Angs")
            #     print(f"Max displacement after optimization: {atoms.get_displacement().max():.4f} Angs")
            
            if args.hessian:
                hessian = calculate_hessian(calc, atoms)
                print(f"Hessian: {hessian}")
            if args.forces:
                forces = calculate_forces(calc, atoms)
                print(f"Forces: {forces}")
            if args.frequencies:
                frequencies = calculate_frequencies(calc, atoms)
                print(f"Frequencies: {frequencies}")

            if args.out_format == 'pybel':
                update_mol(mol, atoms, align=False)
                f.write(mol.write(out_format))
                f.flush()
            elif args.out_format == 'json':
                out_file_name = f"{out_file_base}.json"
                serializable_results = convert_to_serializable(calc.results)
                with open(out_file_name, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(serializable_results, indent=4))
                    f.flush()
            

if __name__ == '__main__':
    main()


