# Compression Benchmark Report


## SimpleCNN_6x2

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**           | N/A         |        98.87% |              5.94 |     67.97 |        15.79 |
| Quantization (Dynamic) | INT8        |        98.84% |              1.72 |      10.6 |         4.97 |
| Quantization (Static)  | INT8        |        98.73% |               N/A |       5.7 |         5.83 |
| Pruning (Unstructured) | 50%         |        98.52% |              2.97 |     22.67 |         6.83 |
| Pruning (Unstructured) | 70%         |        98.36% |              1.78 |     22.67 |         6.72 |
| Pruning (Unstructured) | 30%         |        98.15% |              4.16 |     22.67 |         7.43 |
| Pruning (Structured)   | 30%         |        98.01% |              2.90 |      11.1 |         6.13 |
| Pruning (Structured)   | 50%         |        97.20% |              1.49 |      5.71 |         3.98 |
| Pruning (Structured)   | 70%         |        87.78% |              0.53 |      2.06 |         2.82 |
| Quantization (QAT)     | INT8        |        28.43% |               N/A |      5.73 |         3.77 |

---

## EnhancedLeNet5

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| Pruning (Unstructured) | 50%         |        96.94% |              0.56 |      4.26 |         2.12 |
| Pruning (Unstructured) | 30%         |        96.93% |              0.78 |      4.26 |         1.81 |
| Quantization (Dynamic) | INT8        |        96.90% |              0.05 |      1.23 |         3.47 |
| **Original**           | N/A         |        96.90% |              1.11 |     12.76 |         2.89 |
| Quantization (Static)  | INT8        |        96.51% |               N/A |      1.08 |          2.4 |
| Pruning (Unstructured) | 70%         |        96.35% |              0.33 |      4.26 |         1.36 |
| Pruning (Structured)   | 30%         |        95.44% |              0.54 |      2.06 |         3.06 |
| Pruning (Structured)   | 50%         |        92.20% |              0.28 |      1.08 |         1.79 |
| Pruning (Structured)   | 70%         |        86.23% |              0.10 |      0.39 |         1.69 |
| Quantization (QAT)     | INT8        |        69.52% |               N/A |      1.08 |         2.48 |

---

## EfficientNetB0Custom

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**           | N/A         |        92.19% |              4.06 |     46.98 |       100.78 |
| Quantization (Dynamic) | INT8        |        92.17% |              4.01 |     15.64 |        92.62 |
| Pruning (Unstructured) | 30%         |        91.48% |              2.86 |      15.8 |        77.38 |
| Pruning (Unstructured) | 50%         |        90.14% |              2.06 |      15.8 |        76.05 |
| Pruning (Unstructured) | 70%         |        85.84% |              1.25 |      15.8 |        63.05 |
| Pruning (Structured)   | 30%         |        49.46% |              1.89 |      8.03 |         71.8 |
| Pruning (Structured)   | 50%         |        12.81% |              0.96 |      4.37 |        58.26 |
| Pruning (Structured)   | 70%         |         2.53% |              0.35 |       1.8 |        46.18 |

---

## ResNet50Custom

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| Quantization (Dynamic) | INT8        |        94.55% |             23.51 |     90.07 |      1323.83 |
| **Original**           | N/A         |        94.54% |              23.6 |    270.47 |      1994.86 |
| Pruning (Unstructured) | 50%         |        93.99% |             11.82 |     90.32 |       392.45 |
| Pruning (Unstructured) | 30%         |        93.86% |             16.53 |     90.32 |      1048.56 |
| Pruning (Unstructured) | 70%         |        93.33% |              7.12 |     90.32 |       112.42 |
| Pruning (Structured)   | 30%         |        92.99% |             10.77 |     44.36 |       713.08 |
| Pruning (Structured)   | 50%         |        92.25% |              5.37 |     22.86 |       285.03 |
| Pruning (Structured)   | 70%         |        87.65% |              1.89 |      8.31 |       128.43 |

---
