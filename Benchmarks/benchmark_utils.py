import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import sys
import pandas as pd

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Compression.slimming_utils import physically_prune_structured, make_pruning_permanent
from Compression.metrics_utils import get_input_size

NUM_CLASSES = 43


# 1. Quantization Wrappers & Helpers
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


def load_quantized_model(ModelClass, method, filepath):
    model_name = ModelClass.__name__
    model = ModelClass(num_classes=NUM_CLASSES)

    if method == "Dynamic":
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    elif method in ["Static", "QAT"]:
        model = QuantizedModelWrapper(model)
        model.eval()

        # Replicate Fusing
        try:
            if model_name == "SimpleCNN_6x2":
                torch.ao.quantization.fuse_modules(model.model, [
                    ['conv1', 'bn1'], ['conv2', 'bn2'], ['conv3', 'bn3'],
                    ['conv4', 'bn4'], ['conv5', 'bn5'], ['conv6', 'bn6']
                ], inplace=True)
            elif model_name == "EnhancedLeNet5":
                torch.ao.quantization.fuse_modules(model.model, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
        except Exception:
            pass

        model.qconfig = torch.quantization.get_default_qconfig('x86')
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

    checkpoint = torch.load(filepath, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model



# 2. Pruning Helpers
def apply_pruning_masks(model, pruning_type, amount):
    for name, module in model.named_modules():
        if pruning_type == "unstructured":
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name="weight", amount=amount)
        elif pruning_type == "structured":
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model


def load_pruned_model(ModelClass, pruning_type, amount, filepath):
    model = ModelClass(num_classes=NUM_CLASSES)
    apply_pruning_masks(model, pruning_type, amount)

    checkpoint = torch.load(filepath, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    keys_to_remove = [k for k in state_dict.keys() if "total_ops" in k or "total_params" in k]
    for k in keys_to_remove: del state_dict[k]

    model.load_state_dict(state_dict, strict=False)
    make_pruning_permanent(model)

    if pruning_type == "structured":
        input_size = get_input_size(model)
        model = physically_prune_structured(model, pruning_ratio=amount, example_input_size=input_size)

    return model



# 3. Reporting
def generate_final_report(all_results):
    df = pd.DataFrame(all_results)
    models = df['Model'].unique()

    print("\n\n# Compression Benchmark Report\n")

    for model_name in models:
        print(f"## {model_name}\n")

        model_df = df[df['Model'] == model_name].copy()
        model_df = model_df.sort_values(by='Val. accuracy', ascending=False)

        display_cols = ['Method', 'Compression', 'Val. accuracy', 'Num of params (M)', 'Size (MB)', 'Latency (ms)']

        # Print Markdown
        print(model_df[display_cols].to_markdown(index=False))
        print("\n---\n")