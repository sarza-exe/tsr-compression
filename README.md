# Traffic Sign Recognition (TSR)

> **CNN Model Compression & Real-Time Deployment on Edge Devices**

This project implements an **end-to-end compression and deployment pipeline for Convolutional Neural Networks (CNNs)** applied to traffic sign classification using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

The primary objective is to **reduce model size and inference latency while maintaining high accuracy**, enabling efficient deployment on **resource-constrained edge devices**.

---

## Project Overview

The project covers the **entire lifecycle** of a CNN model:

1. Training baseline (full-precision) models  
2. Applying multiple compression techniques  
3. Benchmarking accuracy, size, and latency  
4. Preparing models for real-time, CPU-efficient deployment  

Compression is performed in a **reproducible and automated pipeline**, allowing fair comparison between architectures and compression strategies.

---

## Supported Architectures

The pipeline supports both custom lightweight models and standard backbones:

- `SimpleCNN_6x2`
- `EnhancedLeNet5`
- `ResNet50Custom`
- `EfficientNetB0Custom`

All models are trained and evaluated on **43 traffic sign classes** from the GTSRB dataset.

---

## Dataset

**GTSRB – German Traffic Sign Recognition Benchmark**

- 43 traffic sign classes  
- RGB images with varying resolutions and lighting conditions  
- Used consistently for:
  - Training teacher (baseline) models
  - Knowledge distillation
  - Validation and benchmarking of compressed models  

The same `train_loader` and `val_loader` are reused across all stages to ensure fair evaluation.

---

## Compression Pipeline

The pipeline applies the following stages **sequentially**:

### 1. Structured Pruning
- Physical (channel/filter-level) pruning
- Removes redundant computation paths
- Produces *slimmed architectures* (no pruning masks)
- Ensures compatibility with quantization

Input models:
```
../Compressed_Models/Pruned_slimmed/*.pt
```

---

### 2. Knowledge Distillation (KD)
- Student: **pruned version of a model**
- Teacher: **unpruned version of the same architecture**
- Distillation recovers accuracy lost during pruning
- Runs on **GPU (CUDA)** when available

Loss combines:
- KL-divergence on softened logits
- Cross-entropy on ground-truth labels

---

### 3. Quantization
- Applied **after distillation**
- Strategy depends on architecture:
  - **Static Quantization (PTQ)**: SimpleCNN, LeNet
  - **Dynamic Quantization**: ResNet, EfficientNet
- Quantization is **CPU-only** (PyTorch limitation)
- Converts FP32 models to INT8 for efficient inference

---

### 4. Evaluation & Benchmarking
For each final model, the pipeline measures:
- Validation accuracy
- Serialized model size (MB)
- CPU inference latency (ms)

These metrics enable direct comparison of accuracy–efficiency trade-offs.

---

## Outputs

### Saved Models
Final compressed models are saved to:
```
../Compressed_Models/Pipeline/
```

Example filename:
```
Final_EfficientNetB0Custom_PrunedS0.7_KD_QuantDynamic.pt
```

This indicates:
- Structured pruning (70%)
- Knowledge distillation
- Dynamic quantization

---

### Results Tables
After processing all models, the pipeline generates a summary table containing:
- Model name
- Applied compression pipeline
- Validation accuracy
- Model size
- Inference latency

The table is:
- Printed to the console
- Saved as:
  - `pipeline_results.csv`
  - `pipeline_results.txt`

Both files are stored alongside the final models.

---

## Deployment

The compressed models are designed for **real-time traffic sign recognition** on edge devices:
- CPU-efficient inference
- Reduced memory footprint
- Suitable for mobile or embedded deployment

A demonstration application supports:
- Live camera input
- Image gallery classification

---

## Device Strategy

- **GPU (CUDA)**:
  - Knowledge distillation training
- **CPU**:
  - Quantization
  - Validation and benchmarking
  - Latency measurement
  - Final model storage and deployment

This separation ensures correctness and avoids unsupported GPU execution of quantized operators.

---

## Project Goals

- Systematically compare compression techniques across architectures
- Quantify trade-offs between accuracy, size, and latency
- Produce deployable CNNs for real-time TSR
- Provide a clean, extensible research and experimentation framework

---

## Summary

This repository provides a **homogeneous, end-to-end CNN compression framework** for traffic sign recognition on the GTSRB dataset, combining **pruning, knowledge distillation, and quantization** into a single automated pipeline with clear metrics and deployable artifacts.
