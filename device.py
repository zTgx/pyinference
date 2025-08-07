import torch

def available_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return device