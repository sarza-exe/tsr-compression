# Mixed Compression Benchmark Report


## SimpleCNN_6x2

| Pipeline                                          |  Val. accuracy | Size (MB) | Latency (ms) |
|:--------------------------------------------------|---------------:|----------:|-------------:|
| **Original**                                      |         98.87% |     67.97 |        15.79 |
| Pruning(unstructured, 0.5) + Quantization(static) |         98.39% |   5.70556 |      24.5858 |
| Pruning(unstructured, 0.7) + Quantization(static) |         98.27% |   5.70556 |      22.5128 |
| Pruning(unstructured, 0.3) + Quantization(static) |         98.04% |   5.70556 |      23.8406 |
| Pruning(structured, 0.3) + Quantization(static)   |         97.92% |   2.80718 |      14.6145 |
| Pruning(structured, 0.5) + Quantization(static)   |         97.05% |   1.45299 |      14.2663 |
| Pruning(structured, 0.7) + Quantization(static)   |         87.51% |  0.535512 |      14.2018 |

---

## EnhancedLeNet5

| Pipeline                                          | Val. accuracy | Size (MB) | Latency (ms) |
|:--------------------------------------------------|--------------:|----------:|-------------:|
| **Original**                                      |        96.90% |     12.76 |         2.89 |
| Pruning(unstructured, 0.5) + Quantization(static) |        96.70% |      1.08 |         5.16 |
| Pruning(unstructured, 0.3) + Quantization(static) |        96.65% |      1.08 |         5.25 |
| Pruning(unstructured, 0.7) + Quantization(static) |        95.86% |      1.08 |         5.04 |
| Pruning(structured, 0.3) + Quantization(static)   |        94.93% |      0.53 |         4.22 |
| Pruning(structured, 0.5) + Quantization(static)   |        91.53% |      0.28 |         4.84 |
| Pruning(structured, 0.7) + Quantization(static)   |        84.71% |      0.11 |         4.73 |

---

## EfficientNetB0Custom

| Pipeline                                           | Val. accuracy | Size (MB) | Latency (ms) |
|:---------------------------------------------------|--------------:|----------:|-------------:|
| **Original**                                       |        92.19% |     46.98 |       100.78 |
| Pruning(unstructured, 0.3) + Quantization(dynamic) |        91.46% |     15.64 |       290.10 |
| Pruning(unstructured, 0.5) + Quantization(dynamic) |        90.15% |     15.64 |        89.19 |
| Pruning(unstructured, 0.7) + Quantization(dynamic) |        85.84% |     15.64 |        60.41 |
| Pruning(structured, 0.3) + Quantization(dynamic)   |        49.47% |      7.93 |       137.23 |
| Pruning(structured, 0.5) + Quantization(dynamic)   |        12.82% |      4.29 |        64.98 |
| Pruning(structured, 0.7) + Quantization(dynamic)   |         2.53% |      1.75 |        37.01 |

---

## ResNet50Custom

| Pipeline                                           | Val. accuracy | Size (MB) | Latency (ms) |
|:---------------------------------------------------|--------------:|----------:|-------------:|
| **Original**                                       |        94.54% |    270.47 |      1994.86 |
| Pruning(unstructured, 0.5) + Quantization(dynamic) |        93.98% |     90.07 |      1361.37 |
| Pruning(unstructured, 0.3) + Quantization(dynamic) |        93.86% |     90.07 |      4183.42 |
| Pruning(unstructured, 0.7) + Quantization(dynamic) |        93.33% |     90.07 |       382.89 |
| Pruning(structured, 0.3) + Quantization(dynamic)   |        92.97% |     44.19 |       1744.8 |
| Pruning(structured, 0.5) + Quantization(dynamic)   |        92.25% |     22.73 |       544.45 |
| Pruning(structured, 0.7) + Quantization(dynamic)   |        87.68% |      8.24 |       440.88 |

---