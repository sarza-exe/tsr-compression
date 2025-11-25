# Unified Compression Benchmark Results: GTSRB

| Model              | Method            | Compression | Val. Acc.  | Complexity (Params/Size) | Latency (ms) |
|:-------------------|:------------------|:------------|:-----------|:-------------------------|:-------------|
| **SimpleCNN_6x2**  | **Original**      | N/A         | **98.87%** | 5.94 M                   | 18.69        |
| SimpleCNN_6x2      | Quant. (Dynamic)  | INT8        | **98.84%** | 10.60 MB                 | 13.28        |
| SimpleCNN_6x2      | Quant. (Static)   | INT8        | **98.73%** | 5.70 MB                  | **5.56**     |
| SimpleCNN_6x2      | Pruning (Unstr.)  | 50%         | **98.52%** | 2.97 M                   | 4.88         |
| SimpleCNN_6x2      | Pruning (Unstr.)  | 70%         | **98.36%** | 1.78 M                   | 4.96         |
| SimpleCNN_6x2      | Pruning (Unstr.)  | 30%         | **98.15%** | 4.16 M                   | 5.38         |
| SimpleCNN_6x2      | Pruning (Struct.) | 30%         | **98.01%** | 2.90 M                   | 4.39         |
| SimpleCNN_6x2      | Pruning (Struct.) | 50%         | **97.20%** | 1.49 M                   | **2.75**     |
| EnhancedLeNet5     | Pruning (Unstr.)  | 50%         | **96.94%** | 0.56 M                   | 7.35         |
| EnhancedLeNet5     | Pruning (Unstr.)  | 30%         | **96.93%** | 0.78 M                   | 7.39         |
| EnhancedLeNet5     | Quant. (Dynamic)  | INT8        | **96.90%** | 1.23 MB                  | 2.12         |
| **EnhancedLeNet5** | **Original**      | N/A         | **96.90%** | 1.11 M                   | 5.63         |
| EnhancedLeNet5     | Quant. (Static)   | INT8        | **96.51%** | 1.08 MB                  | **1.16**     |
| EnhancedLeNet5     | Pruning (Unstr.)  | 70%         | **96.35%** | 0.33 M                   | 7.68         |
| EnhancedLeNet5     | Pruning (Struct.) | 30%         | **95.44%** | 0.54 M                   | 6.74         |
| ResNet50Custom     | Quant. (Dynamic)  | INT8        | **94.55%** | 90.07 MB                 | 559.67       |
| **ResNet50Custom** | **Original**      | N/A         | **94.54%** | 23.60 M                  | 1238.97      |
| ResNet50Custom     | Pruning (Unstr.)  | 50%         | **93.99%** | 11.82 M                  | 394.77       |
| ResNet50Custom     | Pruning (Unstr.)  | 30%         | **93.86%** | 16.53 M                  | 2573.45      |
| ResNet50Custom     | Pruning (Unstr.)  | 70%         | **93.33%** | 7.12 M                   | 125.72       |
| ResNet50Custom     | Pruning (Struct.) | 30%         | **92.98%** | 10.77 M                  | 2048.37      |
| ResNet50Custom     | Pruning (Struct.) | 50%         | **92.25%** | 5.37 M                   | 948.61       |
| **EfficientNetB0** | **Original**      | N/A         | **92.19%** | 4.06 M                   | 152.05       |
| EnhancedLeNet5     | Pruning (Struct.) | 50%         | **92.20%** | 0.28 M                   | 5.10         |
| EfficientNetB0     | Quant. (Dynamic)  | INT8        | **92.17%** | 15.64 MB                 | 79.08        |
| EfficientNetB0     | Pruning (Unstr.)  | 30%         | **91.48%** | 2.86 M                   | 207.82       |
| EfficientNetB0     | Pruning (Unstr.)  | 50%         | **90.14%** | 2.06 M                   | 237.53       |
| SimpleCNN_6x2      | Pruning (Struct.) | 70%         | **87.78%** | 0.53 M                   | 1.80         |
| ResNet50Custom     | Pruning (Struct.) | 70%         | **87.65%** | 1.89 M                   | 371.36       |
| EnhancedLeNet5     | Pruning (Struct.) | 70%         | **86.23%** | 0.10 M                   | 6.35         |
| EfficientNetB0     | Pruning (Unstr.)  | 70%         | **85.84%** | 1.25 M                   | 179.08       |
| EnhancedLeNet5     | Quant. (QAT)      | INT8        | **69.52%** | 1.08 MB                  | 1.23         |
| EfficientNetB0     | Pruning (Struct.) | 30%         | **49.46%** | 1.89 M                   | 112.99       |
| SimpleCNN_6x2      | Quant. (QAT)      | INT8        | **28.43%** | 5.73 MB                  | 6.89         |
| EfficientNetB0     | Pruning (Struct.) | 50%         | **12.81%** | 0.96 M                   | 94.87        |
| EfficientNetB0     | Pruning (Struct.) | 70%         | **2.53%**  | 0.35 M                   | 74.84        |