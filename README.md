# Real-Time Facial Emotion Recognition with Cloud-Optimized Logging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, real-time facial emotion recognition system achieving **80% accuracy on RAF-DB**, **50+ FPS on CPU**, and **~80% storage reduction** through cloud-optimized compression.

## 🎯 Key Features

- **Lightweight Architecture**: SimpleFERCNN with only **1.2M parameters**
- **Real-Time Performance**: 50–60 FPS on a standard CPU (Intel Core i5)
- **Robust Preprocessing**: Adaptive low-light enhancement and pose normalization
- **Cloud-Optimized Logging**: ~80% storage reduction (21.7 KB → 4.3 KB per image)
- **Superior Generalization**: Outperforms ResNet-18 and CBAM on constrained datasets

## 📊 Performance Summary

| Model                   | Accuracy (RAF-DB) | Parameters | Train–Test Gap | Inference Time |
|-------------------------|------------------|------------|----------------|----------------|
| **SimpleFERCNN (Ours)** | **80.0%**        | 1.2M       | 16%            | ~15–20 ms      |
| ResNet-18               | 78.49%           | 11.2M      | 21%            | ~80–100 ms     |
| CBAM-Enhanced CNN       | 78.81%           | 1.3M       | 18%            | ~25–30 ms      |

**Dataset Results**
- FER2013: 62% accuracy
- RAF-DB: 80% accuracy

## 🏗️ System Architecture

Input (640×480) → Face Detection → Adaptive Preprocessing → SimpleFERCNN → Emotion Output → Cloud Logging (WebP + Zlib + Base91)

### Pipeline Components

**Adaptive Preprocessing**
- Low-light enhancement using brightness-aware gamma correction and CLAHE
- Pose normalization using MediaPipe face landmarks (eye-level alignment, yaw–pitch–roll normalization)

**Lightweight CNN (SimpleFERCNN)**
- Input: 48×48 grayscale face image
- 3 convolutional blocks with batch normalization and max-pooling
- Fully connected layers with dropout
- Output: 7 emotion classes (Softmax)

**Cloud-Optimized Compression & Logging**
- Face ROI extraction
- WebP encoding (quality ≈ 80)
- Zlib compression
- Base91 text encoding
- Firebase Realtime Database storage
- Storage reduced from ~21.7 KB to ~4.3 KB per sample

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ritinikhil/Major-FER.git
cd Major-FER
pip install -r requirements.txt
setup firebase cloud 
```

### Requirements

- Python 3.8+
- PyTorch ≥ 1.9.0
- torchvision ≥ 0.10.0
- opencv-python ≥ 4.5.0
- mediapipe ≥ 0.8.9
- firebase-admin ≥ 5.0.0
- numpy ≥ 1.21.0

## 🧠 Training

```bash
# Train on FER2013
python train.py --dataset FER2013 --epochs 50 --batch_size 64

# Train on RAF-DB
python train.py --dataset RAF-DB --epochs 50 --batch_size 64
```

## 🎥 Real-Time Inference

```bash
# Webcam input
python realtime_demo.py --model checkpoints/fercnn_raf.pth

# Video file input
python realtime_demo.py --model checkpoints/fercnn_raf.pth --input path/to/video.mp4

# With cloud logging enabled
python realtime_demo.py --model checkpoints/fercnn_raf.pth --cloud-logging
```

Ensure Firebase credentials are configured via environment variables or a config file.
## Output
<img width="1078" height="635" alt="image" src="https://github.com/user-attachments/assets/c3be9f14-04c6-4894-a56e-8c88e36d3545" />

## 📁 Project Structure

```
Major-FER/
├── models/
│   ├── fercnn.py
│   ├── resnet18.py
│   └── cbam.py
├── preprocessing/
│   ├── low_light.py
│   ├── pose_alignment.py
│   └── face_detection.py
├── compression/
│   ├── roi_extractor.py
│   ├── webp_encoder.py
│   └── base91_encoder.py
├── utils/
│   ├── dataset.py
│   ├── train_utils.py
│   └── firebase_client.py
├── train.py
├── evaluate.py
├── realtime_demo.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## 🔬 Key Findings

SimpleFERCNN (1.2M parameters) achieves **80.0% accuracy on RAF-DB** with a **16% train–test gap**, outperforming deeper baselines such as ResNet-18 and CBAM-enhanced CNNs.

Training separate models for FER2013 and RAF-DB yields significantly better generalization due to domain differences between controlled grayscale images and in-the-wild color images.

Adaptive preprocessing improves robustness under low-light and pose variations without increasing model complexity.

ROI-based compression reduces cloud storage requirements by ~80% while preserving facial expression details.

On a CPU-only setup (Intel Core i5-10400), the system sustains **30+ FPS end-to-end** without GPU acceleration.

## 🎓 Citation

```bibtex
@article{singh2026realtime,
  title={A Lightweight Multi-Model Framework for Robust Facial Emotion Recognition with Cloud-Optimised Logging},
  author={Singh, Nikhil and Dagar, Rohit and Garg, Animesh and Dass, Stephen A},
  journal={Under Review},
  year={2026}
}
```

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or extensions.

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

Nikhil Singh – SRM Institute of Science and Technology  
Rohit Dagar – SRM Institute of Science and Technology  
Animesh Garg – SRM Institute of Science and Technology  
Dr. Stephen Dass A – SRM Institute of Science and Technology  

## 🙏 Acknowledgments

Thanks to the FER2013 and RAF-DB creators, and the PyTorch, MediaPipe, and Firebase communities.

## 📧 Contact

For questions or collaboration, please use the GitHub Issues tab.

⭐ If you find this repository useful, consider giving it a star!
