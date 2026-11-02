# 👗 Fashion Parser 🧥

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![NVIDIA Triton](https://img.shields.io/badge/Triton-Inference--Server-76b900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/nvidia-triton-inference-server)
[![uv](https://img.shields.io/badge/Package--Manager-uv-blue?style=flat&logo=python)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

### 2️⃣ Get the data & configure
Download the [FGVC6 dataset](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data) so it looks like:

```text
data/
├── train.csv
├── label_descriptions.json
└── train/train/*.jpg
```

Then copy the template and adjust it if your data lives elsewhere:
```bash
cp .env.example .env
```

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FGVC6_DATA_SET_ROOT_PATH` | `./data` | Dataset root |
| `PRE_TRAINED_FASHION_WEIGHTS` | _(empty)_ | Optional weights to start from |
| `RANDOM_STATE` | `41` | Seed for the train/val split |
| `NUM_CATS` | `46` | Number of fashion categories (background is added) |
| `IMAGE_SIZE` | `512` | Images/masks are resized to a square of this size |

### 3️⃣ Start Training
```bash
uv run python -m fashion_parser.scripts.train_torch --epochs 10 --batch-size 2
# or, via the installed entry point:
uv run fashion-parser --help
```
Weights are saved to `fashion_mrcnn_pytorch.pth` after every epoch.

### 4️⃣ Run the tests
```bash
uv run --extra dev pytest
```

---

## 🚢 Production Deployment (Triton)

### Export to TorchScript
Prepare your trained model for production:
```bash
uv run python -m fashion_parser.scripts.export_triton --weights fashion_mrcnn_pytorch.pth
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
uv run python -m fashion_parser.scripts.triton_client examples/example_001.jpeg --threshold 0.5
```
Sample output:
```text
Detected 3 objects (score >= 0.50)
  class 32  score 0.947
  class 10  score 0.882
  class 23  score 0.614
```
Or from Python:
```python
from fashion_parser.scripts.triton_client import infer_fashion

result = infer_fashion("examples/example_001.jpeg", server_url="localhost:8000")
result["masks"].shape  # (N, 1, 512, 512)
```
The served model accepts a single `[1, 3, 512, 512]` image per request.

---

## 📁 Project Map

```text
fashion-parser/
├── src/fashion_parser/
│   ├── models/        # PyTorch model definitions & wrappers
│   ├── data/          # Annotation loader & Torch Dataset (RLE decoding)
│   ├── config/        # Environment-driven settings
│   ├── utils/         # Image & RLE helpers
│   └── scripts/       # Training, Exporting & Triton clients
├── tests/             # pytest suite
├── model_repository/  # Triton deployment configurations
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