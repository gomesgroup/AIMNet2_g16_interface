"""
Flask API server for AIMNet2 molecular property predictions.
"""
import torch
from openbabel import pybel
from flask import Flask, request, jsonify
import tempfile
import os
from werkzeug.utils import secure_filename
from io import StringIO
from ase.io import write

from aimnet2ase import AIMNet2Calculator
from aimnet2ase_functions import pybel2atoms, calculate_single_point_energy, calculate_dipole, calculate_charges, optimize, calculate_hessian, calculate_forces

app = Flask(__name__)

# Load the PyTorch model and enable multi-GPU if available
model_path = "/home/gdgomes/AIMNet2_g16_interface/models/aimnet2_wb97m-d3_ens.jpt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.jit.load(model_path, map_location=device)
model.to(device)

# Initialize the calculator with the model
calc = AIMNet2Calculator(model)

def load_and_prepare_input(file_stream):
    """
    Convert the uploaded .xyz file to an ASE Atoms object using Pybel for preprocessing.
    """
    mol = next(pybel.readfile('xyz', file_stream))
    atoms = pybel2atoms(mol)
    return atoms

def run_inference(atoms):
    """
    Run inference using the AIMNet2Calculator and return calculated properties,
    including optimization, Hessian, forces, vibrational frequencies, and final XYZ coordinates.
    """
    calc.do_reset()
    calc.set_charge(0)  # Assuming charge=0 for simplicity; adjust as needed
    atoms.set_calculator(calc)  # Ensure the calculator is set for the atoms object

    # Calculate properties
    energy = calculate_single_point_energy(calc, atoms)
    dipole = calculate_dipole(calc, atoms)
    charges = calculate_charges(calc, atoms)
    # Additional properties
    optimize(atoms)  # Note: This modifies the atoms in-place
    hessian = calculate_hessian(calc, atoms)
    forces = calculate_forces(calc, atoms)
    # frequencies = calculate_frequencies(calc, atoms)  # Uncomment if you want to include frequencies

    # Convert final state of atoms to XYZ format
    xyz_string_io = StringIO()
    write(xyz_string_io, atoms, format="xyz")
    final_xyz = xyz_string_io.getvalue()
    xyz_string_io.close()

    # Prepare the output data
    output_data = {
        "energy": energy,
        "dipole": dipole,
        "charges": charges.tolist(),  # Ensure the data is JSON serializable
        "hessian": hessian.tolist(),
        "forces": forces.tolist(),
        # "frequencies": frequencies.tolist(),  # Uncomment if frequencies are included
        "final_xyz": final_xyz  # Include the final XYZ coordinates
    }
    return output_data

@app.route('/infer', methods=['POST'])
def infer():
    file_storage_object = request.files['file']
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, secure_filename(file_storage_object.filename))
    file_storage_object.save(temp_path)
    
    atoms = load_and_prepare_input(temp_path)
    result = run_inference(atoms)

    os.remove(temp_path)
    os.rmdir(temp_dir)
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
