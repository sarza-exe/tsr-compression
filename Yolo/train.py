from ultralytics import YOLO

def main():
    print("Rozpoczynam ładowanie modelu...")

    # Załaduj wstępnie wytrenowany model YOLOv8n (nano)
    model = YOLO('yolov8n.pt')

    print("Model załadowany. Rozpoczynam trening...")

    # Rozpocznij trening
    results = model.train(
        data='GTSDB_YOLO/data.yaml',

        imgsz=416,  # Rozmiar obrazu 416x416

        epochs=100,
        batch=16,  # Rozmiar batcha (dostosuj do VRAM Twojej karty GPU)
        # Zmniejsz do 8 lub 4, jeśli masz "out of memory" błąd

        name='yolov8n_gtsdb_416',  # Nazwa folderu, gdzie zapiszą się wyniki
        project='runs/detect',  # Folder nadrzędny dla wyników
        exist_ok=True,  # Pozwól na nadpisanie poprzedniego treningu o tej samej nazwie

        device=0  # Użyj pierwszego dostępnego GPU (lub 'cpu', jeśli nie masz GPU)
    )

    print("Trening zakończony.")
    print(f"Najlepszy model został zapisany w: {results.save_dir}/weights/best.pt")


if __name__ == '__main__':
    main()