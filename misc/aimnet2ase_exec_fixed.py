import torch
from aimnet2ase_fixed import AIMNet2Calculator

class AIMNet2Exec:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        try:
            self.model = torch.load(model_path)
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {e}")

    def get_calculator(self, charge=0, **kwargs):
        if self.model is None:
            raise Exception("No model loaded. Please load a model first.")

        calculator = AIMNet2Calculator(self.model, charge, **kwargs)
        return calculator