## Benchmark Report

Models benchmarked against GTSRB validation set.

| Model                | Method       | Pruning % | Val. Acc.  | Params (M) | FLOPs (G) | Latency (ms) |
|:---------------------|:-------------|:----------|:-----------|:-----------|:----------|:-------------|
| SimpleCNN_6x2        | Original     | N/A       | **98.87%** | 5.94M      | 0.18G     | 18.692ms     |
| SimpleCNN_6x2        | Unstructured | 30%       | **98.15%** | 4.16M      | 0.18G     | 5.381ms      |
| SimpleCNN_6x2        | Unstructured | 50%       | **98.52%** | 2.97M      | 0.18G     | 4.876ms      |
| SimpleCNN_6x2        | Unstructured | 70%       | **98.36%** | 1.78M      | 0.18G     | 4.957ms      |
| SimpleCNN_6x2        | Structured   | 30%       | **98.01%** | 2.90M      | 0.09G     | 4.386ms      |
| SimpleCNN_6x2        | Structured   | 50%       | **97.20%** | 1.49M      | 0.04G     | 2.745ms      |
| SimpleCNN_6x2        | Structured   | 70%       | **87.78%** | 0.53M      | 0.02G     | 1.798ms      |
| EnhancedLeNet5       | Original     | N/A       | **96.90%** | 1.11M      | 0.02G     | 5.634ms      |
| EnhancedLeNet5       | Unstructured | 30%       | **96.93%** | 0.78M      | 0.02G     | 7.390ms      |
| EnhancedLeNet5       | Unstructured | 50%       | **96.94%** | 0.56M      | 0.02G     | 7.346ms      |
| EnhancedLeNet5       | Unstructured | 70%       | **96.35%** | 0.33M      | 0.02G     | 7.675ms      |
| EnhancedLeNet5       | Structured   | 30%       | **95.44%** | 0.54M      | 0.01G     | 6.739ms      |
| EnhancedLeNet5       | Structured   | 50%       | **92.20%** | 0.28M      | 0.00G     | 5.102ms      |
| EnhancedLeNet5       | Structured   | 70%       | **86.23%** | 0.10M      | 0.00G     | 6.348ms      |
| ResNet50Custom       | Original     | N/A       | **94.54%** | 23.60M     | 4.13G     | 1238.966ms   |
| ResNet50Custom       | Unstructured | 30%       | **93.86%** | 16.53M     | 4.13G     | 2573.449ms   |
| ResNet50Custom       | Unstructured | 50%       | **93.99%** | 11.82M     | 4.13G     | 394.766ms    |
| ResNet50Custom       | Unstructured | 70%       | **93.33%** | 7.12M      | 4.13G     | 125.717ms    |
| ResNet50Custom       | Structured   | 30%       | **92.98%** | 10.77M     | 2.04G     | 2048.370ms   |
| ResNet50Custom       | Structured   | 50%       | **92.25%** | 5.37M      | 1.07G     | 948.614ms    |
| ResNet50Custom       | Structured   | 70%       | **87.65%** | 1.89M      | 0.40G     | 371.356ms    |
| EfficientNetB0Custom | Original     | N/A       | **92.19%** | 4.06M      | 0.41G     | 152.050ms    |
| EfficientNetB0Custom | Unstructured | 30%       | **91.48%** | 2.86M      | 0.41G     | 207.816ms    |
| EfficientNetB0Custom | Unstructured | 50%       | **90.14%** | 2.06M      | 0.41G     | 237.528ms    |
| EfficientNetB0Custom | Unstructured | 70%       | **85.84%** | 1.25M      | 0.41G     | 179.080ms    |
| EfficientNetB0Custom | Structured   | 30%       | **49.46%** | 1.89M      | 0.22G     | 112.987ms    |
| EfficientNetB0Custom | Structured   | 50%       | **12.81%** | 0.96M      | 0.12G     | 94.872ms     |
| EfficientNetB0Custom | Structured   | 70%       | **2.53%**  | 0.35M      | 0.05G     | 74.843ms     |