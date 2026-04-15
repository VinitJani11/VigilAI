# 👁️ VigilAI: Real-Time Intelligent Surveillance Ecosystem

**Next-Gen YOLOv26 | Biometric Identity | Explainable AI (XAI)**

VigilAI transforms traditional CCTV systems into a proactive, intelligent security platform. By combining the latest YOLOv26 detection model with biometric identification and Explainable AI (XAI), it delivers fast, accurate, and transparent threat detection—eliminating the "black box" limitations of conventional AI systems.

---

## ✨ Key Features

- **Ultra-Low Latency**  
  Achieves **32+ FPS on standard CPUs** — no GPU required.

- **Proactive Threat Detection**  
  Real-time detection of concealed weapons (pistols and knives).

- **Biometric Intelligence**  
  Matches individuals against a secure database and assigns risk levels.

- **Explainable AI (XAI)**  
  Uses Grad-CAM and LIME to justify every alert.

- **Forensic Logging**  
  Logs all detections and biometric matches.

---

## 🛠️ Technology Stack

| Component        | Technology                                  |
|-----------------|----------------------------------------------|
| Vision Engine   | YOLOv26 (NMS-Free Architecture)              |
| Biometrics      | Dlib + face_recognition (128-d embeddings)   |
| Explainable AI  | Grad-CAM + LIME                              |
| Backend         | FastAPI                                      |
| Frontend        | HTML5, Tailwind CSS, JavaScript              |
| Model Format    | ONNX + PyTorch                               |
| Storage         | JSON flat-file database                      |

---

## 📁 Repository Structure

```
VigilAI/
├── main.py
├── best.onnx
├── best.pt
├── live-detection.html
├── explain.html
├── login.html
├── requirements.txt
└── README.md
```

---

## 🚀 Installation Guide

### Prerequisites

- Python 3.9 or 3.10  
- Visual Studio C++ Build Tools (Windows only)

---

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/VigilAI.git
cd VigilAI
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

#### Activate:

**Windows**
```bash
venv\Scripts\activate
```

**macOS/Linux**
```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run Server

```bash
uvicorn main:app --reload
```

---

### 5. Open App

```
http://127.0.0.1:8000/login.html
```

---

## 📊 Deployment Notes

- `best.pt` → Training model  
- `best.onnx` → Production model (**faster CPU inference ~43%**)

✅ Use **best.onnx** for deployment.

---

## ⚖️ Ethical Considerations

VigilAI uses a **Human-in-the-Loop** approach:
- Ensures transparency  
- Reduces bias  
- Allows human verification  

---

## 📌 License

© 2026 VigilAI — Individual Computing Science Project
