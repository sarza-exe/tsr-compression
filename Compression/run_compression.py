import os
import sys
import pandas as pd

# Import compression scripts
from pruning import run_experiment as run_pruning
from quantization import run_experiment as run_quantization
# from distillation import run_experiment as run_distillation

# Import model architectures
from Architectures.cnn_6x2 import SimpleCNN_6x2
from Architectures.enhanced_lenet5 import EnhancedLeNet5
from Architectures.resnet50_custom import ResNet50Custom
from Architectures.efficientnet_b0_custom import EfficientNetB0Custom

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results')))

# List of models to test
models = [SimpleCNN_6x2, EnhancedLeNet5, ResNet50Custom, EfficientNetB0Custom]

# Directory for saving results CSV
RESULTS_DIR = "../Results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "compression_results_1.csv")


# Main loop: run compression for all models
if __name__ == "__main__":
    all_results = []

    for ModelClass in models:
        print(f"\n=== Running compression experiments for model: {ModelClass.__name__} ===")

        # # Run pruning experiment
        # pruning_results = run_pruning(ModelClass)
        # all_results.extend(pruning_results)

        # Run quantization
        try:
            quant_results = run_quantization(ModelClass)
            all_results.extend(quant_results)
        except Exception as e:
            print(f"Skipping quantization for {ModelClass.__name__} due to error: {e}")

        # TODO: implement distillation
        # Run distillation
        # distill_results = run_distillation(ModelClass)
        # all_results.extend(distill_results)

    # Save all results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nAll compression results saved to {CSV_PATH}")