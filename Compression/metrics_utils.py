import os
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

    if model_name == "QuantizedModelWrapper":
        model_name = model.model.__class__.__name__

    return INPUT_SIZES.get(model_name, (1, 3, 32, 32))


def get_file_size_mb(filepath):
    if not os.path.exists(filepath): return 0.0
    return os.path.getsize(filepath) / (1024 * 1024)


def count_params(model):
    return sum(torch.count_nonzero(p).item() for p in model.parameters())


def count_flops(model, device="cuda" if torch.cuda.is_available() else "cpu"):
    input_size = get_input_size(model)
    dummy = torch.randn(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy,))
    return flops


def measure_latency(model, device="cpu", warmup=10, target_time=3.0, min_runs=20):
    """
    Measures latency.
    1. Performs warmup to estimate model speed.
    2. Calculates the number of runs required to last approx 'target_time'.
    3. Guarantees at least 'min_runs' for statistical stability.
    """
    model.to(device)
    model.eval()

    # Dynamic input size handling
    try:
        from Compression.metrics_utils import get_input_size
        input_size = get_input_size(model)
    except (ImportError, NameError):
        input_size = (1, 3, 32, 32)  # Fallback

    if len(input_size) == 3:
        input_size = (1, *input_size)

    dummy = torch.randn(*input_size).to(device)

    # Measure warmup time to estimate model speed
    if device == "cuda":
        torch.cuda.synchronize()

    start_warmup = time.perf_counter()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()
    end_warmup = time.perf_counter()

    # Average time per inference during warmup
    avg_time_warmup = (end_warmup - start_warmup) / warmup

    # CALCULATE NUMBER OF RUNS
    # How many times do we need to run the model to reach e.g. 3.0 seconds?\
    if avg_time_warmup < 1e-6:
        avg_time_warmup = 1e-6

    estimated_runs = int(target_time / avg_time_warmup)

    # Take the LARGER of the two values: calculated vs minimum
    # For ResNet (slow): will take min_runs (e.g., 20) -> test lasts ~20-30s
    # For SimpleCNN (fast): will take estimated (e.g., 2000) -> test lasts ~3s
    runs = max(min_runs, estimated_runs)

    # ACTUAL MEASUREMENT
    # Clean loop without internal checks or overhead
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / runs