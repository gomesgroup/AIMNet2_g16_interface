"""Test suite for the AIMNet2 Flask API server."""
import sys
import pytest

from aimnet2_flask_api_server import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.mark.parametrize("molecule_file", [
    '/home/gdgomes/AIMNet2_g16_interface/examples/H2O.xyz',
    '/home/gdgomes/AIMNet2_g16_interface/examples/CCOC.xyz',
    '/home/gdgomes/AIMNet2_g16_interface/examples/oct.xyz',
    # Add more file paths as needed
])
def test_infer_molecules(client, molecule_file):
    with open(molecule_file, 'rb') as file:
        data = {
            'file': (file, molecule_file.split('/')[-1])
        }
        response = client.post('/infer', content_type='multipart/form-data', data=data)

    assert response.status_code == 200
    json_data = response.get_json()

    # Print the results for each test case
    print(f"\nTesting molecule: {molecule_file.split('/')[-1]}")
    print(f"Response JSON: {json_data}")

    # Check for the presence and type of each key in the response
    assert 'energy' in json_data and isinstance(json_data['energy'], (float, int)), "Energy field is missing or not a number"
    assert 'dipole' in json_data and isinstance(json_data['dipole'], (float, int)), "Dipole field is missing or not a number"
    assert 'charges' in json_data and isinstance(json_data['charges'], list), "Charges field is missing or not a list"
    assert 'hessian' in json_data and isinstance(json_data['hessian'], list), "Hessian field is missing or not a list"
    assert 'forces' in json_data and isinstance(json_data['forces'], list), "Forces field is missing or not a list"
    # assert 'frequencies' in json_data and isinstance(json_data['frequencies'], list), "Frequencies field is missing or not a list"
    assert 'final_xyz' in json_data and isinstance(json_data['final_xyz'], str), "Final XYZ coordinates field is missing or not a string"
    # Optionally, add more detailed checks, such as the shape of the numpy arrays for hessian, forces, and frequencies
    # This could involve converting the lists back to numpy arrays and checking their shapes
    # Example:
    # import numpy as np
    # hessian_array = np.array(json_data['hessian'])
    # assert hessian_array.shape == (expected_shape), "Hessian array shape is incorrect"

