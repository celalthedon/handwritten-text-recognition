<div align="center">

# 🖊️ Handwritten Text Recognition

**Deep Learning pipeline for recognizing handwritten text — from raw image to structured output.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-TrOCR-FFBF00?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/trocr-base-handwritten)
[![Gradio](https://img.shields.io/badge/Gradio-App-F97316?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/CelalIbrahimli/handwritten-text-recognition?style=flat-square&color=yellow)](https://github.com/CelalIbrahimli/handwritten-text-recognition/stargazers)

<br/>

*Built from scratch with CNN + BiLSTM + CTC — plus Microsoft TrOCR for production-grade inference.*

</div>

---

## 📌 Overview

This project delivers a **complete Handwritten Text Recognition (HTR) system** — spanning custom model training, transformer-based OCR inference, and a deployable web application.

Two approaches are implemented and compared:

| Approach | Architecture | Best For |
|---|---|---|
| **Custom Model** | CNN → BiLSTM → CTC | Learning, experiments, fine-tuning |
| **Microsoft TrOCR** | Vision Transformer + Language Model | Production, generalization |

---

## 🗂️ Repository Structure

```
handwritten-text-recognition
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks
│   ├── handwriting_recognition_model.ipynb
│   ├── reading_multiple_lines.ipynb
│   └── reading_multiple_lines_microsoft_model_ipynb.ipynb
│
└── models
    ├── best_model.pt
    └── htr_epoch043_best.pt
```

---

## 🧠 Model Architecture — Custom HTR


```mermaid
flowchart LR
    A[Input Image] --> B[Preprocessing<br/>Resize Normalize Grayscale]
    B --> C[CNN Backbone<br/>Conv + ReLU + Pool]
    C --> D[Feature Map]
    D --> E[Reshape to Sequence]
    E --> F[BiLSTM Layer 1]
    F --> G[BiLSTM Layer 2]
    G --> H[Linear Layer]
    H --> I[CTC Decoder]
    I --> J[Recognized Text]
```


**Why CTC?** Connectionist Temporal Classification allows the model to train without character-level bounding box annotations — only word-level transcriptions are needed.

---

## 📊 Training Results

| Checkpoint | Epoch | CER ↓ |
|---|---|---|
| `htr_epoch043_best.pt` | 43 | **0.1038** |
| `best_model.pt` | — | — |
| `final_last.pt` | — | — |

> **CER (Character Error Rate)** — lower is better. CER = 0.10 means ~90% of characters are correctly recognized.

---

## 🔬 Multi-Line Page Recognition Pipeline

For full-page handwritten documents, a segmentation pipeline is applied before recognition:


```mermaid
flowchart TD
    A[Full Page Image] --> B[Convert to Grayscale]
    B --> C[Thresholding and Denoising]
    C --> D[Horizontal Projection Profile]
    D --> E[Detect Line Regions]
    E --> F[Crop Individual Lines]
    F --> G[Generate Image Variants]
    G --> H[Run TrOCR on Each Variant]
    H --> I[Pick Best Prediction]
    I --> J[Merge Lines]
    J --> K[Final Recognized Text]
```

---

## 🤖 Microsoft TrOCR Integration

Two TrOCR variants are supported via Hugging Face:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Base model (faster)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Large model (higher accuracy)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
```

| Model | Speed | Accuracy | Use Case |
|---|---|---|---|
| `trocr-base-handwritten` | ⚡ Fast | ✅ Good | Real-time apps |
| `trocr-large-handwritten` | 🐢 Slower | 🏆 Best | Offline batch |

---

## 🚀 Gradio Application

An interactive web app for uploading and transcribing handwritten images.

```bash
python app.py
```

**Features:**
- 📤 Upload any handwritten image (JPG, PNG)
- ✂️ Automatic line segmentation with visual preview
- 🔍 Line-by-line OCR using TrOCR
- 📋 Full transcript output
- 💾 Download extracted text as `.txt`

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/CelalIbrahimli/handwritten-text-recognition.git
cd handwritten-text-recognition

# Install dependencies
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision transformers accelerate \
            sentencepiece gradio opencv-python-headless \
            pillow matplotlib numpy
```

---

## 🧪 Usage Guide

### 1. Train the Custom Model

```bash
# Open in Jupyter
jupyter notebook handwriting_recognition_model.ipynb
```

- Configure dataset path and hyperparameters
- Model checkpoints saved automatically
- Evaluate with CER metric

### 2. Run Multi-Line Recognition

```bash
jupyter notebook reading_multiple_lines.ipynb
```

### 3. TrOCR Inference

```bash
jupyter notebook reading_multiple_lines_microsoft_model.ipynb
```

### 4. Web Application

```bash
python app.py
```

---

## 📦 Requirements

```
torch>=2.0.0
torchvision
transformers>=4.30.0
accelerate
sentencepiece
gradio
opencv-python-headless
pillow
matplotlib
numpy
```

---
## ⚠️ Limitations

- Custom model performance depends heavily on handwriting style similarity to training data
- Line segmentation may struggle with very dense or irregular spacing
- Poor image quality (low resolution, shadows, skew) reduces accuracy for all models
- TrOCR generalizes better across unseen handwriting styles

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Celal Ibrahimli**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/celal-ibrahimli-b7a47227b/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/CelalIbrahimli)
[![Instagram](https://img.shields.io/badge/Instagram-Follow-E4405F?style=flat-square&logo=instagram)](https://www.instagram.com/celaalibr/)

---

<div align="center">

**If this project helped you, please consider giving it a ⭐**

*Built with PyTorch · Transformers · OpenCV · Gradio*

</div>
