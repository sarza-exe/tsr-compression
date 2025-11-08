# Instrukcja obsługi
- Ściągnąć FullIJCNN2013.zip ze strony https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html
- Rozpakować i zmienić nazwę na GTSDB_dataset
- Odpalić convertData.py
- Odpalić train.py. Jeśli mamy GPU najlepiej zrobić to na pytorchu dla CUDA.

`pip uninstall torch torchvision`

`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

- Odpalić export.py
- plik gtsdb_yolo_416.ptl zawiera gotowy model na smartfony (Bez NMS)