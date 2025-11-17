import os
import pandas as pd

# Import compression scripts
from pruning import run_experiment as run_pruning
# from compression.quantization import run_experiment as run_quantization
# from compression.distillation import run_experiment as run_distillation

# Import model architectures
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom

# List of models to test
models = [ResNet50Custom, EfficientNetB0Custom]

# Directory for saving results CSV
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "compression_results.csv")


# Main loop: run compression for all models
if __name__ == "__main__":
    all_results = []

    for ModelClass in models:
        print(f"\n=== Running compression experiments for model: {ModelClass.__name__} ===")

        # Run pruning experiment
        pruning_results = run_pruning(ModelClass)
        all_results.extend(pruning_results)

        # TODO: implement quantization
        # Run quantization
        # quant_results = run_quantization(ModelClass)
        # all_results.extend(quant_results)

        # TODO: implement distillation
        # Run distillation
        # distill_results = run_distillation(ModelClass)
        # all_results.extend(distill_results)

    # Save all results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nAll compression results saved to {CSV_PATH}")