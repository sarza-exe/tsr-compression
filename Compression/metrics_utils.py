import torch
import time
from thop import profile

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def count_flops(model, device="cuda" if torch.cuda.is_available() else "cpu", input_size=(1, 3, 32, 32)):
    dummy = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy,))
    return flops

def measure_latency(model, device="cuda" if torch.cuda.is_available() else "cpu", runs=100, input_size=(1, 3, 32, 32)):
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
