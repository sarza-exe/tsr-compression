# Mixed Compression Benchmark Report


## SimpleCNN_6x2

| Method                                              | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                        | N/A         |        98.87% |              5.94 |     67.97 |        15.79 |
| Pruning (Unstructured) | 30%         |        98.15% |              4.16 |     22.67 |         7.43 |
| Pruning (Unstructured) | 50%         |        98.52% |              2.97 |     22.67 |         6.83 |
| Pruning (Unstructured) | 70%         |        98.36% |              1.78 |     22.67 |         6.72 |
| Pruning (unstructured, 0.3) + Quantization (Static) | Mix         |        98.04% |               N/A |      5.71 |         3.92 |
| Pruning (unstructured, 0.5) + Quantization (Static) | Mix         |        98.39% |               N/A |      5.71 |          4.6 |
| Pruning (unstructured, 0.7) + Quantization (Static) | Mix         |        98.27% |               N/A |      5.71 |         4.72 |
| Pruning (Structured)   | 30%         |        98.01% |              2.90 |      11.1 |         6.13 |
| Pruning (Structured)   | 50%         |        97.20% |              1.49 |      5.71 |         3.98 |
| Pruning (Structured)   | 70%         |        87.78% |              0.53 |      2.06 |         2.82 |
| Pruning (structured, 0.3) + Quantization (Static)   | Mix         |        97.91% |               N/A |      2.81 |         2.56 |
| Pruning (structured, 0.5) + Quantization (Static)   | Mix         |        97.05% |               N/A |      1.45 |         2.12 |
| Pruning (structured, 0.7) + Quantization (Static)   | Mix         |        87.51% |               N/A |      0.54 |          2.1 |
| Quantization (Static)  | INT8        |        98.73% |               N/A |       5.70 |         5.83 |
| Quantization (Static) + Pruning (unstructured, 0.3)   | Mix         |        96.45% |               N/A |      5.70 |         5.93 |
| Quantization (Static) + Pruning (unstructured, 0.5)   | Mix         |        88.40% |               N/A |      5.70 |         6.03 |
| Quantization (Static) + Pruning (unstructured, 0.7)   | Mix         |        77.34% |               N/A |      5.70 |         5.73 |

---

## EnhancedLeNet5

| Method                                              | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                        | N/A         |        96.90% |              1.11 |     12.76 |         2.89 |
| Pruning (Unstructured) | 30%         |        96.93% |              0.78 |      4.26 |         1.81 |
| Pruning (Unstructured) | 50%         |        96.94% |              0.56 |      4.26 |         2.12 |
| Pruning (Unstructured) | 70%         |        96.35% |              0.33 |      4.26 |         1.36 |
| Pruning (unstructured, 0.5) + Quantization (Static) | Mix         |        96.70% |               N/A |      1.08 |         1.46 |
| Pruning (unstructured, 0.3) + Quantization (Static) | Mix         |        96.65% |               N/A |      1.08 |         1.82 |
| Pruning (unstructured, 0.7) + Quantization (Static) | Mix         |        95.86% |               N/A |      1.08 |         1.45 |
| Pruning (Structured)   | 30%         |        95.44% |              0.54 |      2.06 |         3.06 |
| Pruning (Structured)   | 50%         |        92.20% |              0.28 |      1.08 |         1.79 |
| Pruning (Structured)   | 70%         |        86.23% |              0.10 |      0.39 |         1.69 |
| Pruning (structured, 0.3) + Quantization (Static)   | Mix         |        94.93% |               N/A |      0.53 |         1.37 |
| Pruning (structured, 0.5) + Quantization (Static)   | Mix         |        91.53% |               N/A |      0.28 |         1.95 |
| Pruning (structured, 0.7) + Quantization (Static)   | Mix         |        84.71% |               N/A |      0.11 |          1.6 |

---

## EfficientNetB0Custom

| Method                                               | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                         | N/A         |        92.19% |              4.06 |     46.98 |       100.78 |
| Pruning (Unstructured) | 30%         |        91.48% |              2.86 |      15.8 |        77.38 |
| Pruning (Unstructured) | 50%         |        90.14% |              2.06 |      15.8 |        76.05 |
| Pruning (Unstructured) | 70%         |        85.84% |              1.25 |      15.8 |        63.05 |
| Pruning (unstructured, 0.3) + Quantization (Dynamic) | Mix         |        91.46% |              2.82 |     15.64 |        55.39 |
| Pruning (unstructured, 0.5) + Quantization (Dynamic) | Mix         |        90.15% |              2.03 |     15.64 |        58.95 |
| Pruning (unstructured, 0.7) + Quantization (Dynamic) | Mix         |        85.84% |              1.24 |     15.64 |        51.43 |
| Pruning (Structured)   | 30%         |        49.46% |              1.89 |      8.03 |         71.8 |
| Pruning (Structured)   | 50%         |        12.81% |              0.96 |      4.37 |        58.26 |
| Pruning (Structured)   | 70%         |         2.53% |              0.35 |       1.8 |        46.18 |
| Pruning (structured, 0.3) + Quantization (Dynamic)   | Mix         |        49.47% |              1.86 |      7.93 |        50.18 |
| Pruning (structured, 0.5) + Quantization (Dynamic)   | Mix         |        12.82% |              0.93 |      4.29 |        40.68 |
| Pruning (structured, 0.7) + Quantization (Dynamic)   | Mix         |         2.53% |              0.33 |      1.75 |        32.51 |

---

## ResNet50Custom

| Method                                               | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                         | N/A         |        94.54% |              23.6 |    270.47 |      1994.86 |
| Pruning (Unstructured) | 30%         |        93.86% |             16.53 |     90.32 |      1048.56 |
| Pruning (Unstructured) | 50%         |        93.99% |             11.82 |     90.32 |       392.45 |
| Pruning (Unstructured) | 70%         |        93.33% |              7.12 |     90.32 |       112.42 |
| Pruning (unstructured, 0.3) + Quantization (Dynamic) | Mix         |        93.86% |             16.47 |     90.07 |        192.4 |
| Pruning (unstructured, 0.5) + Quantization (Dynamic) | Mix         |        93.98% |             11.78 |     90.07 |       108.51 |
| Pruning (unstructured, 0.7) + Quantization (Dynamic) | Mix         |        93.33% |              7.09 |     90.07 |        79.54 |
| Pruning (Structured)   | 30%         |        92.99% |             10.77 |     44.36 |       713.08 |
| Pruning (Structured)   | 50%         |        92.25% |              5.37 |     22.86 |       285.03 |
| Pruning (Structured)   | 70%         |        87.65% |              1.89 |      8.31 |       128.43 |
| Pruning (structured, 0.3) + Quantization (Dynamic)   | Mix         |        92.97% |             10.71 |     44.19 |       562.16 |
| Pruning (structured, 0.5) + Quantization (Dynamic)   | Mix         |        92.25% |              5.32 |     22.73 |       271.91 |
| Pruning (structured, 0.7) + Quantization (Dynamic)   | Mix         |        87.68% |              1.87 |      8.24 |       102.76 |

---