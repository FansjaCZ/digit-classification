import torch

def get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return None
    