import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.mobile_optimizer
import os
from Architectures.enhanced_lenet5 import EnhancedLeNet5


# --- Krok 2: Ustawienia ---

INPUT_PT_FILE = "../Models/EnhancedLeNet5_best.pt"
OUTPUT_PTL_FILE = "./MobileModels/EnhancedLeNet5_best.ptl"
NUM_CLASSES = 43

# Kształt wejściowy: [BatchSize, Channels, Height, Width]
CHANNELS = 3
HEIGHT = 32
WIDTH = 32


# --- Krok 3: Logika Konwersji ---

def convert_model():
    if not os.path.exists(INPUT_PT_FILE):
        print(f"BŁĄD: Nie można znaleźć pliku wejściowego: {INPUT_PT_FILE}")
        print("Upewnij się, że najpierw uruchomiłeś trening i plik .pt został zapisany.")
        return

    print(f"Rozpoczynam konwersję modelu: {INPUT_PT_FILE}")

    # Używamy 'cpu', by zapewnić kompatybilność
    device = torch.device('cpu')

    # 1. Stwórz instancję modelu
    model = EnhancedLeNet5(num_classes=NUM_CLASSES).to(device)

    # 2. Załaduj wagi z checkpointa
    print(f"Ładowanie wag z {INPUT_PT_FILE}...")
    checkpoint = torch.load(INPUT_PT_FILE, map_location=device)

    # Ładujemy tylko 'model_state_dict'
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. Ustaw tryb ewaluacji (WYMAGANE DLA DROPOUT I BATCHNORM!)
    model.eval()

    # 4. Stwórz przykładowe wejście (dummy input)
    example_input = torch.rand(1, CHANNELS, HEIGHT, WIDTH).to(device)
    print(f"Używam przykładowego kształtu wejściowego: {example_input.shape}")

    # 5. Prześledź (trace) model za pomocą TorchScript
    print("Trasowanie modelu (JIT)...")
    traced_module = torch.jit.trace(model, example_input)

    # 6. Optymalizuj dla urządzeń mobilnych (PyTorch Lite)
    print("Optymalizacja dla PyTorch Lite...")
    optimized_lite_module = torch.utils.mobile_optimizer.optimize_for_mobile(traced_module)

    # 7. Zapisz finalny model .ptl
    optimized_lite_module._save_for_lite_interpreter(OUTPUT_PTL_FILE)

    print(f"\n--- SUKCES! ---")
    print(f"Model został pomyślnie przekonwertowany i zapisany jako: {OUTPUT_PTL_FILE}")


if __name__ == "__main__":
    convert_model()