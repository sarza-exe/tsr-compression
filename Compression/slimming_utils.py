import torch
import torch.nn as nn
import torch_pruning as tp
import torch.nn.utils.prune as prune
from typing import Tuple

NUM_CLASSES = 43


# Physically remove channels using torch-pruning (Structured Pruning)
def physically_prune_structured(model: nn.Module, pruning_ratio: float,
                                example_input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> nn.Module:
    model.cpu()
    example_inputs = torch.randn(*example_input_size)

    # Identify classification layer to ignore
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == NUM_CLASSES:
            ignored_layers.append(m)

    imp = tp.importance.MagnitudeImportance(p=1)
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=1,
    )
    pruner.step()
    return model


# Remove pruning masks - make the zero values permanent in the weight tensors
def make_pruning_permanent(model: nn.Module) -> nn.Module:
    for _, module in model.named_modules():
        # Check if the module has active pruning reparametrizations
        if hasattr(module, 'weight_mask'):
            prune.remove(module, 'weight')
    return model
