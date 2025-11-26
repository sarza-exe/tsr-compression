# Compression Benchmark Report


## SimpleCNN_6x2

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**           | N/A         |       98.8678 |              5.94 |     67.97 |        15.79 |
| Quantization (Dynamic) | INT8        |        98.844 |              1.72 |      10.6 |         4.97 |
| Quantization (Static)  | INT8        |       98.7332 |               N/A |       5.7 |         5.83 |
| Pruning (Unstructured) | 50%         |       98.5194 |              2.97 |     22.67 |         6.83 |
| Pruning (Unstructured) | 70%         |        98.361 |              1.78 |     22.67 |         6.72 |
| Pruning (Unstructured) | 30%         |       98.1473 |              4.16 |     22.67 |         7.43 |
| Pruning (Structured)   | 30%         |       98.0127 |              2.90 |      11.1 |         6.13 |
| Pruning (Structured)   | 50%         |       97.1971 |              1.49 |      5.71 |         3.98 |
| Pruning (Structured)   | 70%         |       87.7831 |              0.53 |      2.06 |         2.82 |
| Quantization (QAT)     | INT8        |       28.4323 |               N/A |      5.73 |         3.77 |

---

## EnhancedLeNet5

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| Pruning (Unstructured) | 50%         |       96.9359 |              0.56 |      4.26 |         2.12 |
| Pruning (Unstructured) | 30%         |       96.9279 |              0.78 |      4.26 |         1.81 |
| Quantization (Dynamic) | INT8        |       96.9042 |              0.05 |      1.23 |         3.47 |
| **Original**           | N/A         |       96.9042 |              1.11 |     12.76 |         2.89 |
| Quantization (Static)  | INT8        |       96.5083 |               N/A |      1.08 |          2.4 |
| Pruning (Unstructured) | 70%         |         96.35 |              0.33 |      4.26 |         1.36 |
| Pruning (Structured)   | 30%         |       95.4394 |              0.54 |      2.06 |         3.06 |
| Pruning (Structured)   | 50%         |       92.2011 |              0.28 |      1.08 |         1.79 |
| Pruning (Structured)   | 70%         |       86.2312 |              0.10 |      0.39 |         1.69 |
| Quantization (QAT)     | INT8        |       69.5249 |               N/A |      1.08 |         2.48 |

---

## EfficientNetB0Custom

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| **Original**           | N/A         |       92.1932 |              4.06 |     46.98 |       100.78 |
| Quantization (Dynamic) | INT8        |       92.1694 |              4.01 |     15.64 |        92.62 |
| Pruning (Unstructured) | 30%         |       91.4806 |              2.86 |      15.8 |        77.38 |
| Pruning (Unstructured) | 50%         |       90.1425 |              2.06 |      15.8 |        76.05 |
| Pruning (Unstructured) | 70%         |       85.8353 |              1.25 |      15.8 |        63.05 |
| Pruning (Structured)   | 30%         |       49.4616 |              1.89 |      8.03 |         71.8 |
| Pruning (Structured)   | 50%         |       12.8108 |              0.96 |      4.37 |        58.26 |
| Pruning (Structured)   | 70%         |       2.52573 |              0.35 |       1.8 |        46.18 |

---

## ResNet50Custom

| Method                 | Compression | Val. accuracy | Num of params (M) | Size (MB) | Latency (ms) |
|:-----------------------|:------------|--------------:|------------------:|----------:|-------------:|
| Quantization (Dynamic) | INT8        |       94.5527 |             23.51 |     90.07 |      1323.83 |
| **Original**           | N/A         |       94.5447 |              23.6 |    270.47 |      1994.86 |
| Pruning (Unstructured) | 50%         |       93.9905 |             11.82 |     90.32 |       392.45 |
| Pruning (Unstructured) | 30%         |       93.8559 |             16.53 |     90.32 |      1048.56 |
| Pruning (Unstructured) | 70%         |       93.3254 |              7.12 |     90.32 |       112.42 |
| Pruning (Structured)   | 30%         |        92.985 |             10.77 |     44.36 |       713.08 |
| Pruning (Structured)   | 50%         |       92.2486 |              5.37 |     22.86 |       285.03 |
| Pruning (Structured)   | 70%         |       87.6485 |              1.89 |      8.31 |       128.43 |

---
