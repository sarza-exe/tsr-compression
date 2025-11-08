from ultralytics import YOLO

def validate_model():
    """
    Ten skrypt ładuje wytrenowany model i uruchamia jego walidację
    na zbiorze danych zdefiniowanym w pliku .yaml.
    """

    MODEL_PATH = 'runs/detect/yolov8n_gtsdb_416/weights/best.pt'
    DATA_YAML_PATH = 'GTSDB_YOLO/data.yaml'
    IMAGE_SIZE = 416

    print(f"Ładowanie modelu z: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku modelu w {MODEL_PATH}")
        print("Upewnij się, że ścieżka jest poprawna i model został wytrenowany.")
        return
    except Exception as e:
        print(f"Wystąpił błąd podczas ładowania modelu: {e}")
        return

    print("Rozpoczynam walidację na zbiorze 'val'...")

    # uruchom walidację na wszystkich obrazach z folderu 'images/val'
    metrics = model.val(
        data=DATA_YAML_PATH,  # Plik konfiguracyjny danych
        imgsz=IMAGE_SIZE,  # Rozmiar obrazu musi być taki sam jak przy treningu
        split='val',  # Jawnie określ, że walidacja ma być na zbiorze 'val'
        device=0,  # Użyj GPU (zmień na 'cpu', jeśli jest taka potrzeba)
        save_json=True,  # Zapisz wyniki jako JSON (przydatne do analizy)
        save_hybrid=True  # Zapisz obrazy z etykietami (do wizualnej inspekcji)
    )

    # Wyświetl kluczowe metryki
    print("\n--- Zakończono walidację ---")
    print("Oto kluczowe metryki wydajności modelu:")

    # mAP50 to najważniejsza metryka w większości przypadków detekcji.
    # Oznacza "Mean Average Precision" przy progu IoU (Intersection over Union) > 0.5
    print(f"  mAP50 (IoU=0.50):   {metrics.box.map50 * 100:.2f}%")

    # mAP50-95 to średnia mAP dla progów IoU od 0.50 do 0.95 (co 0.05)
    # Jest to bardziej rygorystyczna metryka.
    print(f"  mAP50-95 (IoU=0.50:0.95): {metrics.box.map * 100:.2f}%")

    # Ponieważ mamy tylko jedną klasę ('sign'),
    # poniższe wartości dotyczą tej jednej klasy.
    print(f"\nMetryki dla klasy 'sign':")
    print(f"  Precyzja (Precision): {metrics.box.p[0] * 100:.2f}%")
    print(f"  Czułość (Recall):     {metrics.box.r[0] * 100:.2f}%")

    print("\nSzczegółowe wyniki, wykresy (np. macierz pomyłek) oraz obrazy z predykcjami")
    print(f"zostały zapisane w folderze: {metrics.save_dir}")


if __name__ == '__main__':
    validate_model()