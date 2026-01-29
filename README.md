# 👗 Fashion Parser 🧥

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![NVIDIA Triton](https://img.shields.io/badge/Triton-Inference--Server-76b900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/nvidia-triton-inference-server)
[![uv](https://img.shields.io/badge/Package--Manager-uv-blue?style=flat&logo=python)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Modern, High-Performance Fashion Image Segmentation.** 
> Restructured, modernized, and ready for production serving.

Fashion segmentation based on **Mask R-CNN** (ResNet-50 FPN) using the [imaterialist-fashion-2019-FGVC6](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/overview) dataset.


---

## 🛠️ Technology Stack

| Category | Technology |
| :--- | :--- |
| **Framework** | PyTorch 2.1+, Torchvision |
| **Backbone** | ResNet-50 Feature Pyramid Network (FPN) |
| **Package Management** | `uv` |
| **Serving** | NVIDIA Triton Inference Server |
| **Configuration** | `python-dotenv` |
| **Visualization** | Matplotlib, OpenCV |

---

## 🏁 Quick Start

### 1️⃣ Installation
Ensure you have [uv](https://github.com/astral-sh/uv) installed, then run:
```bash
uv sync
```

### 2️⃣ Configuration
Copy the template and set your dataset paths:
```bash
cp .env.example .env
```

### 3️⃣ Start Training
```bash
uv run src/fashion_parser/scripts/train_torch.py
```

---

## 🚢 Production Deployment (Triton)

### Export to TorchScript
Prepare your trained model for production:
```bash
uv run src/fashion_parser/scripts/export_triton.py --weights fashion_mrcnn_pytorch.pth
```

### Launch Server
Deploy the model repository using Docker:
```bash
docker run --gpus=all --rm -p 8000:8000 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### Client Inference
```bash
uv run src/fashion_parser/scripts/triton_client.py path/to/image.jpg
```

---

## 📁 Project Map

```text
fashion-parser/
├── src/fashion_parser/
│   ├── models/        # PyTorch model definitions & wrappers
│   ├── data/          # Torch Dataset & Loader engines
│   ├── config/        # Environment-driven settings
│   └── scripts/       # Training, Exporting & Triton clients
├── model_repository/  # Triton deployment configurations
├── scripts/           # Legacy entry points (for reference)
├── old_code/          # Backup of legacy TF/Keras implementation
└── pyproject.toml     # Modern project configuration
```

---

## 🖼️ Example Results

<div align="center">
  <img src="examples/example_001.jpeg?raw=true" width="400" />
  <img src="examples/example_002.jpeg?raw=true" width="400" />
</div>

---

## 📑 Roadmap

- [x] **Phase 1**: Restructure & Modernize Codebase
- [x] **Phase 2**: Migrate to PyTorch & `uv`
- [x] **Phase 3**: Triton Inference Server Integration
- [ ] **Phase 4**: Add Inference & Visualization Jupyter Notebooks