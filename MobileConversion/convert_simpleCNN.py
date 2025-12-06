import torch
import torch.nn as nn
import torch.utils.mobile_optimizer
import os
from Architectures.cnn_6x2 import SimpleCNN_6x2
import torch.nn.utils.prune as prune


# --- Krok 2: Ustawienia ---

INPUT_PT_FILE = "../Models/SimpleCNN_6x2_best.pt"
OUTPUT_PTL_FILE = "./MobileModels/SimpleCNN_6x2_best.ptl"
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
    model = SimpleCNN_6x2(num_classes=NUM_CLASSES).to(device)

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



# --- Krok 4: Logika Konwersji Modelu Pruningowanego ---

PRUNED_INPUT_PT_FILE = "../Compressed_Models/pruned_SimpleCNN_6x2_unstructured_0.7.pt"
OUTPUT_PRUNED_PTL_FILE = "./MobileModels/SimpleCNN_6x2_pruned.ptl"


def convert_pruned_model():
    if not os.path.exists(PRUNED_INPUT_PT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku: {PRUNED_INPUT_PT_FILE}")
        return

    print(f"Rozpoczynam konwersję modelu PRUNINGOWANEGO: {PRUNED_INPUT_PT_FILE}")
    device = torch.device('cpu')

    # 1. Stwórz czystą instancję modelu
    model = SimpleCNN_6x2(num_classes=NUM_CLASSES).to(device)

    # =========================================================================
    # KROK A: PRZYGOTOWANIE MODELU (Musi być PRZED ładowaniem wag!)
    # =========================================================================
    print("Modyfikowanie warstw, aby akceptowały maski (Identity Pruning)...")

    # Iterujemy po wszystkich warstwach. Jeśli warstwa to Conv2d lub Linear,
    # aplikujemy "puste" przycinanie. To zmienia strukturę warstwy tak,
    # że przestaje ona oczekiwać 'weight', a zaczyna oczekiwać 'weight_orig' i 'weight_mask'.
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.identity(module, 'weight')

    # =========================================================================
    # KROK B: CZYSZCZENIE I ŁADOWANIE STATE_DICT
    # =========================================================================
    print(f"Ładowanie pliku {PRUNED_INPUT_PT_FILE}...")
    checkpoint = torch.load(PRUNED_INPUT_PT_FILE, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Usuwanie metadanych, które powodowały poprzedni błąd
    keys_to_remove = ["total_ops", "total_params"]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    # Teraz ładowanie powinno się udać, bo model "oczekuje" kluczy _orig i _mask,
    # a state_dict je "posiada".
    try:
        model.load_state_dict(state_dict)
        print("Wagi załadowane pomyślnie.")
    except RuntimeError as e:
        print("CRITICAL ERROR podczas ładowania wag. Sprawdź nazwy warstw.")
        print(e)
        return

    # =========================================================================
    # KROK C: SCALANIE MASEK (Make Permanent)
    # =========================================================================
    # To jest kluczowe dla mobile. Usuwamy bufory pomocnicze i zostawiamy
    # tylko wynikowy tensor (z dużą ilością zer).
    print("Scalanie masek z wagami (Making Pruning Permanent)...")

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, 'weight')

    # =========================================================================
    # KROK D: KONWERSJA DO MOBILE
    # =========================================================================
    model.eval()

    example_input = torch.rand(1, CHANNELS, HEIGHT, WIDTH).to(device)

    print("Trasowanie modelu (JIT)...")
    traced_module = torch.jit.trace(model, example_input)

    print("Optymalizacja dla PyTorch Lite...")
    optimized_lite_module = torch.utils.mobile_optimizer.optimize_for_mobile(traced_module)

    optimized_lite_module._save_for_lite_interpreter(OUTPUT_PRUNED_PTL_FILE)

    print(f"\n--- SUKCES (PRUNED)! ---")
    print(f"Model zapisany jako: {OUTPUT_PRUNED_PTL_FILE}")



# --- Krok 5: Logika Konwersji Modelu Kwantyzowanego Dynamicznie ---

QUANTIZED_INPUT_PT_FILE = "../Compressed_Models/quantized_SimpleCNN_6x2_Dynamic.pt"
OUTPUT_QUANTIZED_PTL_FILE = "./MobileModels/SimpleCNN_6x2_quantized_dynamic.ptl"


def convert_dynamic_quantized_model():
    if not os.path.exists(QUANTIZED_INPUT_PT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku: {QUANTIZED_INPUT_PT_FILE}")
        return

    print(f"Rozpoczynam konwersję modelu KWANTYZOWANEGO: {QUANTIZED_INPUT_PT_FILE}")
    device = torch.device('cpu')

    # 1. Stwórz czystą instancję modelu (Float32)
    float_model = SimpleCNN_6x2(num_classes=NUM_CLASSES).to(device)

    # =========================================================================
    # KROK A: ODTWORZENIE STRUKTURY KWANTYZACJI
    # =========================================================================
    # Musimy przekształcić model Float32 w model Quantized Dynamic,
    # zanim spróbujemy załadować wagi. Jeśli tego nie zrobimy,
    # PyTorch nie będzie wiedział, gdzie włożyć spakowane wagi int8.

    print("Konwersja struktury modelu do Dynamic Quantized...")

    # UWAGA: Zakładamy, że kwantyzowane były warstwy liniowe (nn.Linear),
    # co jest standardem dla kwantyzacji dynamicznej.
    quantized_model = torch.quantization.quantize_dynamic(
        float_model,
        {nn.Linear},  # Określamy, które warstwy mają być podmienione
        dtype=torch.qint8
    )

    # =========================================================================
    # KROK B: CZYSZCZENIE I ŁADOWANIE STATE_DICT
    # =========================================================================
    print(f"Ładowanie wag z {QUANTIZED_INPUT_PT_FILE}...")
    checkpoint = torch.load(QUANTIZED_INPUT_PT_FILE, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Usuwanie metadanych (total_ops etc.)
    keys_to_remove = ["total_ops", "total_params"]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    try:
        quantized_model.load_state_dict(state_dict)
        print("Wagi załadowane pomyślnie.")
    except RuntimeError as e:
        print("CRITICAL ERROR: Struktura modelu nie pasuje do wag.")
        print("Upewnij się, że używasz tych samych parametrów kwantyzacji co podczas kompresji.")
        print(e)
        return

    # =========================================================================
    # KROK C: KONWERSJA DO MOBILE
    # =========================================================================
    quantized_model.eval()

    # Dummy input (nadal float! W dynamicznej kwantyzacji wejście jest float,
    # a konwersja do int8 dzieje się w locie wewnątrz warstwy).
    example_input = torch.rand(1, CHANNELS, HEIGHT, WIDTH).to(device)

    print("Trasowanie modelu (JIT)...")
    traced_module = torch.jit.trace(quantized_model, example_input)

    print("Optymalizacja dla PyTorch Lite...")
    optimized_lite_module = torch.utils.mobile_optimizer.optimize_for_mobile(traced_module)

    optimized_lite_module._save_for_lite_interpreter(OUTPUT_QUANTIZED_PTL_FILE)

    print(f"\n--- SUKCES (DYNAMIC QUANTIZED)! ---")
    print(f"Model zapisany jako: {OUTPUT_QUANTIZED_PTL_FILE}")


if __name__ == "__main__":
    convert_model()
    convert_pruned_model()
    convert_dynamic_quantized_model()