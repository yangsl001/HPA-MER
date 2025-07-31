# src/models/model_main.py
from .hpa_mer_model import HPAMerModel
from .baseline_models import MulT


def get_model(config):
    model_name = config.MODEL_NAME.lower()
    if model_name == 'hpa_mer':
        return HPAMerModel(config)
    elif model_name == 'mult':
        return MulT(config)
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")
# Factory function to get model instance by name
