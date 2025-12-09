# Traffic Sign Recognition (TSR)

> **CNN Model Compression & Real-Time Deployment on Edge Devices**

## Project Overview
This project focuses on optimizing Convolutional Neural Networks (CNNs) for the classification of traffic signs using the **GTSRB** dataset. The main goal is to reduce model size and inference latency while maintaining high accuracy.

The project covers the entire pipeline: from training baseline models, through applying advanced compression techniques (**Pruning** and **Quantization**), to deploying and testing the models in a real-time **mobile demonstration application**.

## Key Features
* **Architectures:** Implementation and optimization of custom CNNs (e.g., SimpleCNN, LeNet) and standard backbones (ResNet, EfficientNet).
* **Compression Techniques:**
    * **Pruning:** Structured and Unstructured pruning to remove redundant weights.
    * **Quantization:** Dynamic, Static (PTQ), and Quantization-Aware Training (QAT) to convert models from FP32 to INT8.
* **Deployment:** A demo application allowing real-time traffic sign classification from a camera or image gallery.

## Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)

## Benchmarks
We evaluate the trade-offs between **accuracy**, **model size**, and **CPU latency** to find the optimal configuration for edge deployment.
