# 🚦 Traffic Sign Detection - Android Edge AI

An Android application built with **Kotlin** that performs real-time traffic sign detection using an edge-optimized **YOLOv8** model via **PyTorch Mobile**. 

This project demonstrates a complete End-to-End Machine Learning pipeline: from dataset preprocessing and custom model training to on-device mobile deployment.

## ✨ Key Features
* **Real-Time Detection:** Processes camera frames on the edge without the need for an internet connection.
* **Custom Trained Model:** Utilizes a YOLOv8-nano model trained specifically on the German Traffic Sign Detection Benchmark (GTSDB).
* **Optimized for Mobile:** The model is exported to PyTorch Lite (`.ptl`) with integrated Non-Maximum Suppression (NMS) for low-latency inference on mobile CPUs.

## 🛠️ Tech Stack & Tools
* **Android / UI:** Kotlin, Android SDK, CameraX
* **Edge AI Inference:** PyTorch Mobile (LiteNativePeer)
* **Model Training:** Python, PyTorch, Ultralytics YOLOv8
* **Data Processing:** OpenCV, Pandas (Dataset normalization and conversion)

## 🧠 The Machine Learning Pipeline
1. **Data Preparation:** Normalized the GTSDB dataset images (1360x800) and converted bounding box annotations into the YOLO format.
2. **Training:** Fine-tuned `yolov8n.pt` for 100 epochs, tracking mAP50 to ensure high detection accuracy.
3. **Export & Optimization:** Traced the model using TorchScript, wrapped it with custom normalization layers, and optimized it for mobile (`optimize_for_mobile`) to generate the final `.ptl` artifact.
4. **Deployment:** Integrated the lightweight model into the Android app assets, utilizing Kotlin coroutines for smooth background inference without blocking the UI thread.

## Screenshots
![screenshot](https://github.com/user-attachments/assets/6ba479f2-4549-489d-bcc0-cabf3cf1fc54)
