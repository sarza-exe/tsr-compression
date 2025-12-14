# Mixed Compression Benchmark Report


## SimpleCNN_6x2

| Method                                              | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                        | N/A         |        98.87% |              5.94 |     67.97 |        15.79 |
| Pruning (unstructured, 0.5) + Quantization (Static) | Mix         |        98.39% |               N/A |      5.71 |          4.6 |
| Pruning (unstructured, 0.7) + Quantization (Static) | Mix         |        98.27% |               N/A |      5.71 |         4.72 |
| Pruning (unstructured, 0.3) + Quantization (Static) | Mix         |        98.04% |               N/A |      5.71 |         3.92 |
| Pruning (structured, 0.3) + Quantization (Static)   | Mix         |        97.91% |               N/A |      2.81 |         2.56 |
| Pruning (structured, 0.5) + Quantization (Static)   | Mix         |        97.05% |               N/A |      1.45 |         2.12 |
| Pruning (structured, 0.7) + Quantization (Static)   | Mix         |        87.51% |               N/A |      0.54 |          2.1 |

---

## EnhancedLeNet5

| Method                                              | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                        | N/A         |        96.90% |              1.11 |     12.76 |         2.89 |
| Pruning (unstructured, 0.5) + Quantization (Static) | Mix         |        96.70% |               N/A |      1.08 |         1.46 |
| Pruning (unstructured, 0.3) + Quantization (Static) | Mix         |        96.65% |               N/A |      1.08 |         1.82 |
| Pruning (unstructured, 0.7) + Quantization (Static) | Mix         |        95.86% |               N/A |      1.08 |         1.45 |
| Pruning (structured, 0.3) + Quantization (Static)   | Mix         |        94.93% |               N/A |      0.53 |         1.37 |
| Pruning (structured, 0.5) + Quantization (Static)   | Mix         |        91.53% |               N/A |      0.28 |         1.95 |
| Pruning (structured, 0.7) + Quantization (Static)   | Mix         |        84.71% |               N/A |      0.11 |          1.6 |

---

## EfficientNetB0Custom

| Method                                               | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                         | N/A         |        92.19% |              4.06 |     46.98 |       100.78 |
| Pruning (unstructured, 0.3) + Quantization (Dynamic) | Mix         |        91.46% |              2.82 |     15.64 |        55.39 |
| Pruning (unstructured, 0.5) + Quantization (Dynamic) | Mix         |        90.15% |              2.03 |     15.64 |        58.95 |
| Pruning (unstructured, 0.7) + Quantization (Dynamic) | Mix         |        85.84% |              1.24 |     15.64 |        51.43 |
| Pruning (structured, 0.3) + Quantization (Dynamic)   | Mix         |        49.47% |              1.86 |      7.93 |        50.18 |
| Pruning (structured, 0.5) + Quantization (Dynamic)   | Mix         |        12.82% |              0.93 |      4.29 |        40.68 |
| Pruning (structured, 0.7) + Quantization (Dynamic)   | Mix         |         2.53% |              0.33 |      1.75 |        32.51 |

---

## ResNet50Custom

| Method                                               | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------------------------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**                                         | N/A         |        94.54% |              23.6 |    270.47 |      1994.86 |
| Pruning (unstructured, 0.5) + Quantization (Dynamic) | Mix         |        93.98% |             11.78 |     90.07 |       108.51 |
| Pruning (unstructured, 0.3) + Quantization (Dynamic) | Mix         |        93.86% |             16.47 |     90.07 |        192.4 |
| Pruning (unstructured, 0.7) + Quantization (Dynamic) | Mix         |        93.33% |              7.09 |     90.07 |        79.54 |
| Pruning (structured, 0.3) + Quantization (Dynamic)   | Mix         |        92.97% |             10.71 |     44.19 |       562.16 |
| Pruning (structured, 0.5) + Quantization (Dynamic)   | Mix         |        92.25% |              5.32 |     22.73 |       271.91 |
| Pruning (structured, 0.7) + Quantization (Dynamic)   | Mix         |        87.68% |              1.87 |      8.24 |       102.76 |

---