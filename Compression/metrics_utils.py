import torch
import time
from thop import profile

INPUT_SIZES = {
    "SimpleCNN_6x2": (1, 3, 32, 32),
    "EnhancedLeNet5": (1, 3, 32, 32),
    "ResNet50Custom": (1, 3, 224, 224),
    "EfficientNetB0Custom": (1, 3, 224, 224)
}

def get_input_size(model):
    model_name = model.__class__.__name__
    input_size = INPUT_SIZES[model_name]
    return input_size

def count_params(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters())

def count_flops(model, device="cuda" if torch.cuda.is_available() else "cpu"):
    input_size = get_input_size(model)
    dummy = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy,))
    return flops

def measure_latency(model, device="cuda" if torch.cuda.is_available() else "cpu", runs=100):
    input_size = get_input_size(model)
    model.to(device)
    model.eval()

    dummy = torch.randn(*input_size).to(device)

    for _ in range(10):
        _ = model(dummy)

    start = time.time()
    for _ in range(runs):
        _ = model(dummy)
    end = time.time()

    return (end - start) / runs
