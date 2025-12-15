import os
import sys
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.ao.quantization
from torch.ao.quantization import fuse_modules
import copy

# Path Setup (Matching your structure)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Architectures
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5

# Import Helpers
from data_original import val_loader
from metrics_utils import measure_latency, get_file_size_mb

# --- 1. THE HACK: Forceful Pruning Function ---
def forceful_prune_quantized_layer(module, amount=0.5):
    """
    Unpacks a quantized layer, zeroes out 'amount' proportion of weights,
    and repacks them.
    Handles the structural difference between Linear (Wrapper) and Conv2d (Raw).
    """
    if not isinstance(module, (nnq.Linear, nnq.Conv2d)):
        return module

    # --- A. UNPACKING ---
    if isinstance(module, nnq.Linear):
        # Linear uses a Wrapper. 
        # We access the raw C++ object inside the wrapper to unpack.
        # Note: module._packed_params is the wrapper.
        weight_q, bias = module._packed_params._weight_bias()
    else:
        # Conv2d stores the C++ object directly.
        weight_q, bias = module._packed_params.unpack()
    
    # --- B. ACCESS INT8 REPR ---
    weight_int = weight_q.int_repr()
    
    # --- C. CREATE MASK ---
    # --- C. CREATE MASK (L1 Magnitude) ---
    # 1. Calculate the "real" magnitude relative to the zero-point
    #    (A value of 0 might actually be -128 in real math if zero_point is 128)
    if weight_q.qscheme() == torch.per_tensor_affine:
        z = weight_q.q_zero_point()
        # Magnitude is distance from zero_point
        magnitude = torch.abs(weight_int.float() - z)
        
    elif weight_q.qscheme() == torch.per_channel_affine:
        # Get zero points and broadcast them
        z_points = weight_q.q_per_channel_zero_points()
        axis = weight_q.q_per_channel_axis()
        shape_broadcast = [1] * weight_int.ndim
        shape_broadcast[axis] = -1
        z_expanded = z_points.view(shape_broadcast).expand_as(weight_int)
        
        magnitude = torch.abs(weight_int.float() - z_expanded.float())
    else:
        # Fallback
        magnitude = torch.abs(weight_int.float())

    # 2. Find the threshold value for the bottom X%
    #    flatten() puts all weights in one long line to find the cutoff
    threshold = torch.quantile(magnitude.flatten(), amount)

    # 3. Create mask: Keep weights strictly greater than threshold
    mask = magnitude > threshold
    
    # --- D. APPLY MASK (Zeroing out) ---
    pruned_int = weight_int.clone()
    
    # Check Scheme and Apply Zero Points
    if weight_q.qscheme() == torch.per_tensor_affine:
        z_point = int(weight_q.q_zero_point())
        pruned_int[~mask] = z_point
        
        new_weight_q = torch._make_per_tensor_quantized_tensor(
            pruned_int, weight_q.q_scale(), weight_q.q_zero_point()
        )
        
    elif weight_q.qscheme() == torch.per_channel_affine:
        scales = weight_q.q_per_channel_scales()
        z_points = weight_q.q_per_channel_zero_points()
        axis = weight_q.q_per_channel_axis()
        
        shape_broadcast = [1] * pruned_int.ndim
        shape_broadcast[axis] = -1
        z_points_expanded = z_points.view(shape_broadcast).expand_as(pruned_int)
        
        pruned_int[~mask] = z_points_expanded[~mask].to(pruned_int.dtype)
        
        new_weight_q = torch._make_per_channel_quantized_tensor(
            pruned_int, scales, z_points, axis
        )
    else:
        return module

    # --- E. RE-PACKING & ASSIGNMENT (THE FIX) ---
    if isinstance(module, nnq.Linear):
        # 1. Create the new C++ PackedParams object
        new_packed_params = torch.ops.quantized.linear_prepack(new_weight_q, bias)
        
        # 2. Assign it INSIDE the existing wrapper
        # module._packed_params is the Python Wrapper
        # module._packed_params._packed_params is the C++ Object we need to replace
        object.__setattr__(module._packed_params, "_packed_params", new_packed_params)
        
    elif isinstance(module, nnq.Conv2d):
        # 1. Create the new C++ PackedParams object
        new_packed_params = torch.ops.quantized.conv2d_prepack(
            new_weight_q, bias, module.stride, module.padding, 
            module.dilation, module.groups
        )
        
        # 2. Assign it DIRECTLY to the module (replacing the old C++ object)
        object.__setattr__(module, "_packed_params", new_packed_params)

    return module

def apply_forceful_pruning(model, amount=0.5):
    """Iterates over the model and prunes every quantized layer found."""
    print(f"   -> Forcefully pruning {amount*100}% of weights in quantized layers...")
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, (nnq.Linear, nnq.Conv2d)):
            forceful_prune_quantized_layer(module, amount)
            count += 1
    print(f"      Pruned {count} layers.")
    return model

# --- 2. RECONSTRUCTION HELPER ---
# We must recreate the EXACT architecture structure to load the state_dict
class QuantizedModelWrapper(nn.Module):
    def __init__(self, model_to_wrap):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model_to_wrap
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def load_quantized_model_structure(ModelClass, method="Static"):
    """
    Replays the fusion and conversion logic from quantization.py
    to create a skeleton that matches the saved state_dict.
    """
    # 1. Init clean float model
    model = ModelClass(num_classes=43)
    
    # 2. Wrap
    if method in ["Static", "QAT"]:
        model = QuantizedModelWrapper(model)
        
    model.cpu()
    model.eval()

    # 3. REPLAY FUSION (Must match quantization.py exactly)
    # If we don't fuse, the keys (e.g., 'conv1') won't match 'conv1.weight' in state_dict
    inner_model = model.model if hasattr(model, 'model') else model
    model_name = ModelClass.__name__
    
    try:
        if model_name == "SimpleCNN_6x2":
            fuse_modules(inner_model, [
                ['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],
                ['conv4', 'bn4'], ['conv5', 'bn5'], ['conv6', 'bn6']
            ], inplace=True)
        elif model_name == "EnhancedLeNet5":
            fuse_modules(inner_model, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
    except Exception:
        pass # If fusion fails, we proceed (assuming the saved model wasn't fused either)

    # 4. PREPARE & CONVERT (Dummy pass to switch layer types)
    # We don't need calibration data here, we just need the structure conversion
    if method == "Static":
        model.qconfig = torch.quantization.get_default_qconfig('x86')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
    elif method == "QAT":
        model.qconfig = torch.quantization.get_default_qat_qconfig('x86')
        torch.quantization.prepare_qat(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        
    return model

# --- 3. MAIN EXPERIMENT ---
if __name__ == "__main__":
    # Settings
    MODEL_CLASS = SimpleCNN_6x2 # Change this to test other models
    METHOD = "Static"           # Or "QAT"
    PRUNING_AMOUNT = 0.7        # 50% sparsity
    
    MODEL_NAME = MODEL_CLASS.__name__
    LOAD_PATH = f"../Compressed_Models/Quantization/quantized_{MODEL_NAME}_{METHOD}.pt"
    
    print(f"=== Experiment: Pruning {METHOD} Quantized {MODEL_NAME} ===")
    
    if not os.path.exists(LOAD_PATH):
        print(f"Error: File {LOAD_PATH} not found. Run quantization.py first.")
        sys.exit(1)

    # 1. Load the Skeleton
    print("1. Reconstructing Quantized Architecture...")
    q_model = load_quantized_model_structure(MODEL_CLASS, METHOD)
    
    # 2. Load Weights
    print(f"2. Loading weights from {LOAD_PATH}...")
    state_dict = torch.load(LOAD_PATH, map_location="cpu")
    q_model.load_state_dict(state_dict)
    
    # 3. Baseline Validation
    print("3. Validating Baseline (Before Pruning)...")
    criterion = nn.CrossEntropyLoss()
    # We need to import validate from your train_utils or define a simple loop
    # Assuming standard signature from your code:
    from Training.train_utils import validate
    _, base_acc = validate(q_model, val_loader, criterion, torch.device("cpu"))
    base_lat = measure_latency(q_model, device="cpu")
    print(f"   [Baseline] Acc: {base_acc:.4f} | Latency: {base_lat*1000:.2f} ms")

    # 4. FORCE PRUNING
    print("\n4. Applying Forceful Pruning...")
    pruned_model = apply_forceful_pruning(q_model, amount=PRUNING_AMOUNT)
    
    # 5. Validate Pruned Model
    print("5. Validating Pruned Model...")
    _, pruned_acc = validate(pruned_model, val_loader, criterion, torch.device("cpu"))
    pruned_lat = measure_latency(pruned_model, device="cpu")
    
    # 6. Check File Size (It won't change)
    save_path = "temp_pruned_quantized.pt"
    torch.save(pruned_model.state_dict(), save_path)
    size_mb = get_file_size_mb(save_path)
    os.remove(save_path)
    
    print("\n=== RESULTS ===")
    print(f"Pruning Amount: {PRUNING_AMOUNT*100}%")
    print(f"Accuracy Drop:  {base_acc:.4f} -> {pruned_acc:.4f} (Delta: {pruned_acc-base_acc:.4f})")
    print(f"Latency Change: {base_lat*1000:.2f}ms -> {pruned_lat*1000:.2f}ms")
    print(f"File Size:      {size_mb:.2f} MB (Unchanged because sparse Int8 is stored densely)")
    print("Conclusion: Pruning a quantized model degrades accuracy with NO performance gain.")