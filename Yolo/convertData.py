import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# GTSDB_YOLO/
# ├── images/
# │ ├── train/ -- obrazki jpeg do treningu (720) w rozmiarach 1360x800
# │ └── val/ -- jpeg to walidacji (180)
# ├── labels/ -- współrzędne wszystkich znakow na obrazku
# │ ├── train/
# │ └── val/
# └── data.yaml -- plik konfiguracyjny

# ścieżka do folderu, z plikiem gt.txt i obrazami .ppm
INPUT_DIR = Path('GTSDB_dataset')

GT_FILE = INPUT_DIR / 'gt.txt'

# Nazwa folderu wyjściowego
OUTPUT_DIR = Path('GTSDB_YOLO')

TRAIN_VAL_SPLIT_RATIO = 0.8

print(f"Tworzenie struktury folderów w: {OUTPUT_DIR}")
(OUTPUT_DIR / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

try:
    # Definiujemy nazwy kolumn dla czytelności
    col_names = ['filename', 'x1', 'y1', 'x2', 'y2', 'class_id']
    data = pd.read_csv(GT_FILE, sep=';', names=col_names)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku {GT_FILE}.")
    print("Upewnij się, że zmienna INPUT_DIR jest poprawnie ustawiona.")
    exit()

grouped = data.groupby('filename')

# Sortujemy unikalne nazwy plików, aby podział train/val był deterministyczny
unique_filenames = sorted(data['filename'].unique())
total_images = len(unique_filenames)
split_index = int(total_images * TRAIN_VAL_SPLIT_RATIO)

print(f"Znaleziono {total_images} obrazów. Podział: {split_index} (train) / {total_images - split_index} (val).")
print("Rozpoczynanie konwersji...")

for i, filename in enumerate(tqdm(unique_filenames)):

    is_train = i < split_index
    split_folder = 'train' if is_train else 'val'

    base_filename = Path(filename).stem  # np. '00000'
    input_img_path = INPUT_DIR / filename

    # Konwertujemy obrazy do .jpg dla wygody i mniejszego rozmiaru
    output_img_path = OUTPUT_DIR / 'images' / split_folder / f"{base_filename}.jpg"
    output_label_path = OUTPUT_DIR / 'labels' / split_folder / f"{base_filename}.txt"

    try:
        with Image.open(input_img_path) as img:
            img_w, img_h = img.size
            # Zapisz jako JPG (konwersja z PPM)
            img.convert('RGB').save(output_img_path, 'JPEG')
    except FileNotFoundError:
        print(f"\nOSTRZEŻENIE: Pominięto. Nie znaleziono obrazu: {input_img_path}")
        continue
    except Exception as e:
        print(f"\nBŁĄD podczas przetwarzania obrazu {input_img_path}: {e}")
        continue

    # Przetwarzanie etykiet Bounding Boxów
    boxes = grouped.get_group(filename)

    yolo_labels = []
    for _, row in boxes.iterrows():
        # Wczytaj współrzędne z pliku gt.txt
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']

        # Konwersja na format YOLO (środek_x, środek_y, szerokość, wysokość)
        box_w = x2 - x1
        box_h = y2 - y1
        x_center = x1 + (box_w / 2)
        y_center = y1 + (box_h / 2)

        # Normalizacja względem wymiarów obrazu
        x_center_norm = x_center / img_w
        y_center_norm = y_center / img_h
        width_norm = box_w / img_w
        height_norm = box_h / img_h

        # class_id = 0 (dla pojedynczej klasy "sign")
        class_id = 0

        # Dodaj sformatowaną linię
        yolo_labels.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")

    # Zapisz plik .txt z etykietami YOLO
    with open(output_label_path, 'w') as f:
        f.write('\n'.join(yolo_labels))

# Utworzenie pliku data.yaml
yaml_content = f"""
# Plik konfiguracyjny datasetu YOLOv8 dla GTSDB (1 klasa)

# Ścieżki do danych (względne do TEGO pliku .yaml)
train: ./images/train
val: ./images/val

# Liczba klas
nc: 1

# Nazwy klas
names: ['sign']
"""

yaml_path = OUTPUT_DIR / 'data.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("\n--- Zakończono! ---")
print(f"Pomyślnie utworzono dataset w formacie YOLO w folderze: {OUTPUT_DIR.resolve()}")
print(f"Plik konfiguracyjny został zapisany jako: {yaml_path.resolve()}")
print("\nMożesz teraz przejść do Kroku 2: Treningu modelu, używając pliku data.yaml.")