# AI-Powered Unattended Baggage Detection System 🛡️🎒

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://github.com/yourusername/Unattended-Baggage-Detection/actions/workflows/main.yml/badge.svg)](https://github.com/yourusername/ai-baggage-detection/actions)
[![Hugging Face](https://img.shields.io/badge/YOLOv5-Object%20Detection-red)](https://github.com/ultralytics/yolov5)

A real-time surveillance system leveraging **YOLOv5** and **CNNs** to detect unattended baggage in high-security environments like airports. Developed in collaboration with the **Airport Authority of India (AAI)**.

![Demo1](demo/demo1.png)  
*Figure 1: Real-time detection dashboard with threat metrics and controls*

---

## 🚨 Critical Notice  
**This system supplements — but does not replace — human security oversight.**  
Immediately report confirmed threats to airport authorities using designated protocols.

---

## 🏆 Key Features
- **Real-Time Detection**  
  🎯 YOLOv10-powered object detection (95.2% mAP)  
  🕒 Temporal analysis for abandoned baggage (5-minute threshold)
- **Automated Security Integration**  
  🔔 REST API alerts to CCTV control rooms  
  📡 WebSocket integration for live video feeds
- **Advanced Analytics**  
  📊 Crowd density heatmaps & threat probability scoring  
  📈 Historical incident logging with SQL/NoSQL support
- **Enterprise-Grade UI**  
  🖥️ Multi-camera grid view with ROI configuration  
  🛠️ Role-based access control (Admin/Operator/Auditor)

---

## 🧰 Technologies Used
| Category              | Tools/Frameworks                          |
|-----------------------|-------------------------------------------|
| **AI/ML**             | YOLOv5, TensorFlow, OpenCV, Scikit-learn  |
| **Backend**           | FastAPI, Redis, PostgreSQL, RabbitMQ     |
| **DevOps**            | Docker, Kubernetes, GitHub Actions       |
| **Monitoring**        | Grafana, Prometheus, Elastic Stack       |
| **Hardware**          | NVIDIA Jetson, Intel OpenVINO, ONNX Runtime |

---

## 📦 Installation

### Prerequisites
- NVIDIA GPU with CUDA 11.2+
- Python 3.8+
- Airport CCTV network access (RTSP/ONVIF)

1. **Clone with submodules**:
```bash
git clone --recurse-submodules https://github.com/SahilPatil-codes/Unattended-Baggage-Detection.git
cd Unattended-Baggage-Detection
