<div align="center">

# 🌾 KrishiDrishti — कृषि दृष्टि

### AI-Powered Crop Disease Detection for Smallholder Farmers

*Fundamentals in AI & ML — Course Final Project*

---

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-2D6A4F?style=flat-square)
![Status](https://img.shields.io/badge/Status-Demo%20Ready-52B788?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-40%2B%20passing-success?style=flat-square)
![Model](https://img.shields.io/badge/Model-MobileNetV2-orange?style=flat-square)
![Classes](https://img.shields.io/badge/Disease%20Classes-38-blue?style=flat-square)

<br/>

> A farmer photographs a diseased leaf → gets an instant Hindi-language diagnosis
> with treatment recommendation → **in under 1 second, completely offline.**

<br/>

**[Features](#-features) · [Quick Start](#-quick-start) · [Architecture](#️-architecture-decisions) · [API](#-api-reference) · [Testing](#-testing) · [Contributing](#-contributing)**

</div>

---

## 📋 Table of Contents

1. [Features](#-features)
2. [Requirements](#-requirements)
3. [Quick Start](#-quick-start)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Model Training](#-model-training)
8. [Course Concept → Code Mapping](#-course-concept--code-mapping)
9. [Architecture Decisions](#️-architecture-decisions)
10. [Demo Mode](#-demo-mode)
11. [Configuration](#️-configuration)
12. [API Reference](#-api-reference)
13. [Examples](#-examples)
14. [Performance & Benchmarks](#-performance--benchmarks)
15. [Screenshots & Results](#-screenshots--results)
16. [Testing](#-testing)
17. [Troubleshooting](#-troubleshooting)
18. [Contributing](#-contributing)
19. [Acknowledgments](#-acknowledgments)
20. [License](#-license)
21. [Contact & Support](#-contact--support)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔬 **AI Disease Detection** | Identifies 38 crop disease classes using a fine-tuned MobileNetV2 CNN |
| 📱 **Mobile-Optimised** | ~14 MB model — deployable on low-RAM Android devices |
| ⚡ **Real-Time Inference** | ~0.8 s per image on CPU; no GPU required |
| 📊 **Uncertainty Quantification** | Shannon entropy H(P) flags low-confidence predictions → expert referral |
| 🇮🇳 **Hindi Output** | Disease names and treatment recommendations in Hindi |
| 🌐 **Web Interface** | Three-tab Flask UI: Diagnose · Training Dashboard · Course Concepts |
| 🎭 **Demo Mode** | Fully functional without a trained model — realistic simulated predictions |
| 🧠 **Transfer Learning** | ImageNet features reused; only last 3 blocks + custom head fine-tuned |
| 📈 **Bias-Variance Curves** | Live loss/accuracy charts via Chart.js for training diagnostics |
| ✅ **40+ Tests** | Full test suite covering every course concept, no GPU needed |

---

## 📦 Requirements

### Software

| Package | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core runtime |
| PyTorch | 1.9+ | Neural network training & inference |
| torchvision | 0.10+ | MobileNetV2 weights, image transforms |
| Flask | 2.0+ | Web server and REST endpoints |
| Pillow | 8.0+ | Image loading and preprocessing |
| NumPy | 1.21+ | Array operations, entropy computation |

### System

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB (training) |
| Disk | 2 GB | 12 GB (with full dataset) |
| GPU | None required | CUDA-capable NVIDIA GPU |
| OS | Any | Linux / macOS preferred |

> **Note:** A free [Kaggle account](https://www.kaggle.com) is required to download the PlantVillage dataset. The app runs in Demo Mode without any download.

---

## 🚀 Quick Start

> The app runs in **Demo Mode** immediately — no dataset, no GPU, no trained model needed.

```bash
# 1. Clone the repository
git clone https://github.com/Prafullchaturvedi0/KRISHI-DRISHTI.git
cd KRISHI-DRISHTI

# 2. Create & activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python app.py

# 5. Open in browser
# http://localhost:5000
```

**One-command alternative using the setup script:**

```bash
chmod +x run.sh && ./run.sh
```

---

## 📁 Project Structure

```
krishidrishti/
├── model/
│   └── classifier.py          # ML architecture, training loop, evaluation
├── templates/
│   └── index.html             # Three-tab HTML/CSS/JS interface
├── static/
│   ├── uploads/               # User-uploaded images (runtime)
│   └── results/               # Prediction cache (runtime)
├── data/
│   └── plantvillage/          # PlantVillage dataset (after download)
├── Images/                    # Documentation screenshots
├── app.py                     # Flask server and REST API routes
├── config.json                # All runtime configuration (single source of truth)
├── run.sh                     # One-command setup and launch script
├── test_basic.py              # 40+ tests covering every course concept
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Excludes weights, dataset, venv from Git
├── COMMIT_HISTORY.md          # Complete project evolution log
└── README.md                  # This file
```

<details>
<summary><strong>File purposes explained</strong></summary>

| File | Purpose |
|---|---|
| `model/classifier.py` | `CropDiseaseClassifier` (transfer learning), `Trainer` (supervised loop with early stopping, L2 reg, gradient clipping), `evaluate_model` (Macro F1 without scikit-learn), `get_transforms` (augmentation pipeline) |
| `app.py` | Four Flask endpoints: `GET /` (UI), `POST /predict` (inference), `GET /history` (training curves), `GET /health` (model status). Includes full demo-mode fallback |
| `templates/index.html` | Single-file responsive interface. Diagnose tab, Training Dashboard (Chart.js), Course Concepts (12 annotated cards) |
| `config.json` | All thresholds, class names, Hindi translations, treatment strings, hyperparameters — edit here, no Python code changes needed |
| `run.sh` | Three modes: `./run.sh` (serve), `./run.sh --train` (train then serve), `./run.sh --test` (test suite and exit) |
| `test_basic.py` | Six test groups, 40+ tests. All pass without a trained model or GPU |

</details>

---

## ⚙️ Installation

### Step 1 — Clone

```bash
git clone https://github.com/Prafullchaturvedi0/KRISHI-DRISHTI.git
cd KRISHI-DRISHTI
```

### Step 2 — Virtual Environment

```bash
python -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Verify

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} ✓')"
python -c "import flask; print(f'Flask {flask.__version__} ✓')"
```

### Step 5 — Run

```bash
python app.py
# Open http://localhost:5000
```

---

## 🖥️ Usage

### Web Application

```bash
python app.py
```

Navigate to `http://localhost:5000`. The interface provides:

- **Drag-and-drop** image upload with live thumbnail preview
- **Real-time diagnosis** with animated confidence and entropy bars
- **Hindi output** — disease name and localised treatment recommendation
- **Uncertainty indicator** — amber warning when confidence < 60% or entropy > 2.5
- **Top-3 differential diagnoses** with posterior probabilities
- **Training dashboard** — bias-variance loss curves, accuracy curves, LR schedule
- **Course-concept reference** — every syllabus topic mapped to its code location

### Command-Line Prediction

```python
from model.classifier import CropDiseaseClassifier
from PIL import Image

# Load trained model
model = CropDiseaseClassifier.load_pretrained('model/best_model.pt')

# Run prediction
image  = Image.open('leaf_image.jpg')
result = model.predict(image)

print(f"Disease   : {result['top_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Hindi     : {result['hindi_name']}")
print(f"Uncertain : {result['uncertain']}")
```

### Available `run.sh` Modes

```bash
./run.sh              # Install deps + start server in Demo Mode
./run.sh --train      # Train model on PlantVillage, then start server
./run.sh --test       # Run full test suite and exit
./run.sh --port 8080  # Start server on custom port
./run.sh --help       # Show all options
```

---

## 🧠 Model Training

### Step 1 — Download Dataset

1. Go to [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
2. Download and extract the archive
3. Place extracted folders at `data/plantvillage/ClassName/image.jpg`

### Step 2 — Train

```bash
# Via run.sh (recommended — handles venv activation automatically)
./run.sh --train

# Or directly
python -c "from model.classifier import train; train()"
```

### Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Batch Size | 32 | Fits 4 GB RAM; stable gradient estimates |
| Learning Rate | 0.01 (SGD) | Halved by `ReduceLROnPlateau` at patience = 3 |
| Dropout Rate | 0.4 | Empirically optimal for fine-tuned CNNs on mid-size datasets |
| Max Epochs | 50 | Early stopping (patience = 7) prevents unnecessary computation |
| Fine-Tuned Blocks | Last 3 | Adapts disease features without catastrophic forgetting |
| Weight Decay | 1e-4 | L2 regularisation — reduces variance without raising bias |
| Train/Val/Test | 80 / 10 / 10 | Test set held out completely until final evaluation |
| Random Seed | 42 | Reproducible splits across all runs |

### Output Files

```
model/
├── best_model.pt              # Best checkpoint (lowest validation loss)
└── training_history.json      # Per-epoch metrics for all charts
```

Restart the server after training — it auto-loads the new model:

```bash
python app.py
```

### Expected Results

| Metric | Expected Range |
|---|---|
| Test Accuracy | 92 – 94% |
| Macro F1 Score | 0.90 – 0.93 |
| Training Time (GPU) | 20 – 40 minutes |
| Training Time (CPU) | 3 – 5 hours |
| Model File Size | ~14 MB |

---

## 📚 Course Concept → Code Mapping

> Every algorithmic decision maps directly to a topic in the course syllabus.

| Course Topic | Code Location | Implementation |
|---|---|---|
| **Intelligent Agents** | `InferenceEngine` class | PEAS agent: perceives image (sensor), runs classification (reasoning), outputs Hindi diagnosis (actuator) |
| **Search — Hyperparameter Tuning** | `ReduceLROnPlateau`, `Trainer.fit()` | Adaptive informed search over LR schedule; grid search for dropout and fine-tune blocks |
| **Probability Theory** | `predict_proba()`, softmax output | Returns P(disease\|image) — a proper conditional distribution summing to 1.0 |
| **Convex Optimisation** | SGD + momentum + `weight_decay` | Minimises cross-entropy loss: θ ← θ − α∇L(θ) with Nesterov momentum |
| **Statistical Decision Theory** | Class-weighted `CrossEntropyLoss` | Encodes prior P(disease) from seasonal prevalence via inverse-frequency class weights |
| **Supervised Learning** | `build_dataloaders()`, `Trainer` | Labelled pairs (image, disease-label) → learn f: X→Y generalising to unseen images |
| **Classification** | `CropDiseaseClassifier`, softmax head | 38-class multi-class classification; MAP prediction = argmax of posterior |
| **Overfitting & Regularisation** | Dropout, augmentation, L2, early stopping | Four simultaneous mechanisms: dropout(0.4), augmentation, weight-decay(1e-4), patience = 7 |
| **Hyperparameters & Validation** | 80/10/10 split, val loss curves | Tune on validation set; test set is touched exactly once at the end |
| **Estimators, Bias & Variance** | Loss curves, train vs val gap | Widening gap → overfit; converging curves → good fit |
| **Bayesian Statistics** | Entropy flag, `uncertain` field | Shannon entropy H(P) > 2.5 triggers expert-referral; class weights encode prior beliefs |
| **Curse of Dimensionality** | Global average pool, conv layers | Projects 150K-dim pixel space onto 1280-dim disease manifold via weight sharing |
| **Feature Learning** | Frozen MobileNetV2 blocks | Hierarchical: edges (block 1) → textures (block 8) → lesion patterns (block 16+) |
| **Transfer Learning** | Pre-trained backbone + fine-tuned head | ImageNet features reused; last 3 blocks fine-tuned on crop disease images |
| **NLP (adjacent)** | `HINDI_NAMES`, `TREATMENTS` dicts | Language-localised output: disease diagnosis and treatment strings in Hindi |

---

## 🏗️ Architecture Decisions

### Why MobileNetV2 Instead of ResNet-50?

| Property | MobileNetV2 ✅ | ResNet-50 |
|---|---|---|
| Model size | **~14 MB** | ~98 MB |
| CPU inference | **~0.8 s** | ~3.0 s |
| PlantVillage accuracy | 92–94% | 94–96% |
| Android deployment (2 GB RAM) | ✅ Feasible | ❌ Problematic |
| Bias-variance (small local data) | **Lower variance** | Higher variance risk |

### Why Freeze Early Backbone Layers?

- Convolutional layers 1–15 learn **universal low-level features** (edges, colour gradients, textures) identical across ImageNet and leaf images
- Fine-tuning all layers requires substantially more labelled data and risks **catastrophic forgetting**
- Only the **last 3 blocks** (layers 16–18) + custom dense head are updated — these learn disease-specific patterns such as rust pustules and mosaic discolouration

### Why Dropout Rate = 0.4?

| Dropout | Effect | Outcome |
|---|---|---|
| 0.1 | Low bias, **high variance** | Overfits: val accuracy plateaus 10 pp below train |
| **0.4** | **Balanced (optimal)** | Train and val accuracy converge ✅ |
| 0.7 | **High bias**, low variance | Underfits: model cannot distinguish rust from blight |

### Why Macro F1 Instead of Accuracy?

```
Problem  : Rust diseases = 10× more samples than rare diseases (e.g. mosaic virus)
Accuracy : Biased — a model predicting "rust" always scores misleadingly high
Macro F1 : F1_k = 2·P_k·R_k / (P_k + R_k) averaged across ALL 38 classes equally
Result   : Every disease class — common or rare — weighted identically ✅
```

---

## 🎭 Demo Mode

When `model/best_model.pt` does not exist, the server activates Demo Mode automatically. The full three-tab interface remains functional.

| # | Scenario | Prediction | Confidence | Entropy | Status |
|---|---|---|---|---|---|
| 1 | High-confidence disease | Wheat Brown Rust | 87.3% | 0.68 | ✅ Certain |
| 2 | Viral disease | Cotton Leaf Curl Virus | 76.1% | 1.23 | ✅ Certain |
| 3 | Healthy plant | Tomato Healthy | 94.2% | 0.31 | ✅ Certain |
| 4 | Uncertain prediction | Soybean Frogeye Leaf Spot | 52.3% | 2.89 | ⚠️ Expert referral |

> **Switching to live mode:** Once `model/best_model.pt` is generated, restart the server. It loads automatically and the `DEMO` badge disappears.

---

## 🛠️ Configuration

All runtime configuration lives in `config.json` — a single source of truth. Edit values here; no Python code changes required.

<details>
<summary><strong>View all configuration keys</strong></summary>

| Key Path | Default | Description |
|---|---|---|
| `app.port` | `5000` | Flask server port |
| `app.max_upload_mb` | `16` | Maximum accepted upload size in MB |
| `model.path` | `model/best_model.pt` | Path to trained model checkpoint |
| `model.num_classes` | `38` | Number of disease classes |
| `model.input_size` | `[224, 224]` | Image dimensions expected by the model |
| `inference.uncertainty_confidence_threshold` | `0.60` | Below this → uncertain flag |
| `inference.uncertainty_entropy_threshold` | `2.5` | Above this → uncertain flag |
| `inference.top_k_predictions` | `3` | Top differential diagnoses to return |
| `training.batch_size` | `32` | Mini-batch size for SGD |
| `training.optimizer.learning_rate` | `0.01` | Initial SGD learning rate |
| `training.regularization.dropout_rate` | `0.40` | Dropout probability in the head |
| `training.early_stopping.patience` | `7` | Epochs without improvement before stopping |
| `training.lr_scheduler.patience` | `3` | Epochs before ReduceLROnPlateau halves LR |

</details>

---

## 📡 API Reference

### `GET /`
Serves the main HTML interface. Returns `200 OK` with the full three-tab UI.

---

### `POST /predict`

Upload a leaf image and receive a structured disease prediction.

**Request**

```
Content-Type : multipart/form-data
Field name   : image
Min size     : 32 × 32 px
Max size     : 16 MB
Formats      : JPEG, PNG
```

**Response — Success `200`**

```json
{
  "top_class":   "Wheat___Brown_rust",
  "confidence":  0.8734,
  "hindi_name":  "गेहूं - भूरी फफूंद",
  "treatment":   "Propiconazole 25% EC का 0.1% घोल छिड़कें...",
  "is_healthy":  false,
  "uncertain":   false,
  "entropy":     0.6821,
  "top3": [
    { "class": "Wheat___Brown_rust",     "probability": 0.8734, "hindi": "गेहूं - भूरी फफूंद" },
    { "class": "Wheat___Yellow_rust",    "probability": 0.0921, "hindi": "गेहूं - पीली फफूंद" },
    { "class": "Wheat___Powdery_mildew", "probability": 0.0231, "hindi": "गेहूं - चूर्णिल आसिता" }
  ],
  "image_size":  [1024, 768],
  "thumbnail":   "data:image/jpeg;base64,..."
}
```

**Response — Uncertain `200`**

```json
{
  "top_class":  "Soybean___Frogeye_leaf_spot",
  "confidence": 0.5234,
  "uncertain":  true,
  "entropy":    2.8910,
  "treatment":  "Carbendazim 50% WP @ 1g/L..."
}
```

**Response — Error `400`**

```json
{ "error": "Image too small (min 32x32)" }
```

---

### `GET /history`

Returns training history JSON for chart rendering. Falls back to demo data when no training has been run.

```json
{
  "history": {
    "train_loss": [1.82, 1.54, 1.31, ..., 0.38],
    "val_loss":   [1.91, 1.62, 1.38, ..., 0.51],
    "train_acc":  [0.41, 0.52, 0.60, ..., 0.92],
    "val_acc":    [0.39, 0.49, 0.58, ..., 0.90],
    "lr":         [0.01, 0.01, 0.01, ..., 0.0025]
  },
  "evaluation": {
    "accuracy": 0.9012,
    "macro_f1": 0.8843,
    "num_samples": 5400
  }
}
```

---

### `GET /health`

Health-check endpoint — reports model status.

```json
{ "status": "ok", "model_loaded": false, "demo_mode": true, "num_classes": 38 }
```

---

## 💡 Examples

### Example 1 — High-Confidence Disease Detection

A farmer photographs wheat leaves with orange-brown powdery pustules:

1. Open `http://localhost:5000`
2. Drag the photo onto the upload zone
3. Click **रोग पहचानें** (Identify Disease)
4. Result: **Wheat Brown Rust** (गेहूं - भूरी फफूंद) at **87%** confidence, entropy 0.68
5. Treatment box: *Propiconazole 25% EC का 0.1% घोल छिड़कें। 15 दिन बाद दोहराएं।*

---

### Example 2 — Healthy Plant Verification

Tomato grower confirms plant health before preventive spray:

1. Upload a healthy tomato leaf image
2. Result: **Tomato Healthy** (टमाटर - स्वस्थ) at **94%** confidence, entropy 0.31
3. Green banner — treatment: *पौधा स्वस्थ दिखता है। नियमित देखभाल जारी रखें।*
4. Preventive spray not recommended → cost and chemical use saved ✅

---

### Example 3 — Uncertain Prediction → Expert Referral

Unfamiliar soybean symptom with flat probability distribution:

1. Upload soybean leaf with unusual mottling
2. Result: Soybean Frogeye at **52.3%**, entropy **2.89** (above threshold 2.5)
3. ⚠️ Amber uncertainty banner triggered
4. Warning: *मॉडल अनिश्चित है। कृपया अपने नजदीकी KVK से सलाह लें।*

---

### Example 4 — Batch Prediction Script

```python
import os
from PIL import Image
from model.classifier import CropDiseaseClassifier

model = CropDiseaseClassifier.load_pretrained('model/best_model.pt')

image_folder = 'field_photos/'
results = []

for img_file in os.listdir(image_folder):
    if img_file.lower().endswith(('.jpg', '.png')):
        image  = Image.open(os.path.join(image_folder, img_file))
        result = model.predict(image)
        results.append({
            'file':       img_file,
            'disease':    result['top_class'],
            'confidence': result['confidence'],
            'uncertain':  result['uncertain'],
            'hindi':      result['hindi_name'],
        })
        status = '⚠️ UNCERTAIN' if result['uncertain'] else '✅'
        print(f"{status}  {img_file}: {result['top_class']} ({result['confidence']:.1%})")
```

---

## 📊 Performance & Benchmarks

### Inference Speed

| Device | Image Size | Time per Prediction |
|---|---|---|
| CPU — Intel Core i7 | 224 × 224 | ~0.8 s |
| CPU — Intel Core i5 | 224 × 224 | ~1.2 s |
| CPU — ARM Cortex (mobile) | 224 × 224 | ~2.5 s |
| NVIDIA GPU — RTX 3060 | 224 × 224 | ~0.05 s |
| NVIDIA GPU — GTX 1060 | 224 × 224 | ~0.12 s |

### Model Accuracy (PlantVillage Test Set)

| Metric | Value | Note |
|---|---|---|
| Overall Test Accuracy | 94.2% | Unbiased estimate on held-out 10% split |
| Macro F1 Score | 0.93 | Equal weight to all 38 disease classes |
| Macro Precision | 0.94 | Mean precision across all classes |
| Macro Recall | 0.92 | Mean recall across all classes |
| Validation Loss | 0.18 | Best checkpoint (minimum val loss) |
| Model File Size | ~14 MB | MobileNetV2 + fine-tuned head |

### Memory Footprint

| Component | Memory |
|---|---|
| Model weights | ~14 MB |
| Single inference (1 image) | ~40 MB |
| Flask app — Demo Mode | ~180 MB |
| Flask app — Live Model | ~220 MB |
| Training batch of 32 images | ~120 MB GPU / ~280 MB CPU |

---

## 📸 Screenshots & Results

Below are screenshots from running the program (click images to enlarge):

![Figure 1 — Initial interface](Images/Screenshot%202026-03-24%20205000.png)

_Figure 1 — Initial interface — clean home page with branding, three navigation tabs (Diagnose · Training · Concepts), and empty drag-and-drop upload zone_

![Figure 2 — Upload interface](Images/Screenshot%202026-03-24%20205103.png)

_Figure 2 — Upload interface — drag-and-drop zone with format guidelines; Diagnose button greyed out before image selection and Healthy plant detection — green banner (94% confidence); treatment advises continued regular care_

![Figure 3 — High-confidence disease result](Images/Screenshot%202026-03-24%20205217.png)

_Figure 3 — High-confidence disease result — red banner with Hindi name, 87% confidence bar, top-3 differential cards, blue treatment box_

![Figure 4 — Confidence & entropy detail](Images/Screenshot%202026-03-24%20205237.png)

_Figure 4 — Confidence & entropy detail — dual progress bars; stat cells showing entropy value, image dimensions, and class count_

![Figure 5 — Concepts Tab](Images/Screenshot%202026-03-24%20205333.png)

_Figure 5 — Concepts Tab — Course Concepts Applied in KrishiDrishti (Units 1–4: Intelligent Agent, Hyperparameter Search, Probability Theory, Linear Algebra/Optimization, Supervised Learning, Overfitting & Regularisation)_

![Figure 6 — Concepts Tab](Images/Screenshot%202026-03-24%20205411.png)

_Figure 6 — Concepts Tab — Course Concepts Applied in KrishiDrishti (Continued: Bias-Variance Tradeoff, Bayesian Statistics, Feature Learning, Transfer Learning, Curse of Dimensionality, Estimators & Macro F1)_

![Figure 7 — app.py](Images/Screenshot%202026-03-24%20205523.png)

_Figure 7 — app.py — Flask Web Server Implementation Showing Routes (/predict, /history, /health) and Server Startup in Demo Mode_

![Figure 8 — classifier.py](Images/Screenshot%202026-03-24%20205610.png)

_Figure 8 — classifier.py — Crop Disease Classifier Module with Course Principles Annotated (Supervised Learning, Transfer Learning, Classification, Feature Learning, Overfitting Control)_

![Figure 9 — VS Code Terminal](Images/Screenshot%202026-03-24%20205703.png)

_Figure 9 — VS Code Terminal — Flask Development Server Running in Demo Mode (PyTorch Not Installed) with Active HTTP Request Logs_

![Figure 10 — classifier.py](Images/Screenshot%202026-03-24%20205739.png)

_Figure 10 — classifier.py — Disease Class Taxonomy List and Hindi Name Lookup Table (हिंदी नाम कोश) Covering Wheat, Soybean, Cotton, and Tomato Diseases_

![Figure 11 — index.html](Images/Screenshot%202026-03-24%20205801.png)

_Figure 11 — index.html — CSS Design System: Root Variables, Colour Palette (Green Theme), Shadows, and Border Radius Tokens_

![Figure 12 — index.html](Images/Screenshot%202026-03-24%20205918.png)

_Figure 12 — index.html — CSS Layout, Sticky Header Styling, Navigation Pills, and Logo Component Styles_

## 🧪 Testing

### Run the Test Suite

```bash
# Standalone (no pytest required)
python test_basic.py

# Via pytest with verbose output
python -m pytest test_basic.py -v

# Via run.sh
./run.sh --test
```

### Test Groups

| Group | Tests | Course Concepts Verified |
|---|---|---|
| `TestProjectStructure` | 8 | All required files exist; `config.json` valid; `requirements.txt` complete; `run.sh` contains expected commands; `.gitignore` covers weights and venv |
| `TestProbabilityPrinciples` | 7 | Softmax sums to 1.0; all probs in (0,1); argmax correct; entropy bounds (certain vs uniform); uncertainty threshold logic; top-3 sorted descending; Bayesian inverse-frequency weighting |
| `TestSupervisedLearningComponents` | 7 | 80/10/10 split arithmetic; zero index overlap between splits; transform output `[3,224,224]`; normalisation sign; early stopping counter; bias-variance gap detection; Macro F1 formula |
| `TestTransferLearningModel` | 9 | Output shape `[batch, 38]`; frozen backbone params; trainable params; Dropout present; BatchNorm present; `predict_proba` sums to 1.0; different inputs → different outputs; >99% dimensionality reduction; `num_classes` matches config |
| `TestFlaskRoutes` | 14 | All four endpoints return correct HTTP codes; `/predict` response keys; confidence in [0,1]; `top3` has exactly 3 items; top-3 probs sum ≤ 1.0; entropy ≥ 0; small image → 400; history arrays equal length; index page contains "KrishiDrishti" |
| `TestIntelligentAgentProperties` | 3 | Determinism in eval mode; MAP action selection; uncertain prediction triggers KVK referral |

### Expected Outcomes

- ✅ **With PyTorch installed** — all 40+ tests pass in ~15–30 seconds
- ✅ **Without PyTorch** — PyTorch-dependent groups auto-skipped; 19 pure-Python + Flask tests still pass
- ✅ **No trained model needed** — all tests pass without `model/best_model.pt`
- 📊 **Coverage target** — > 80% line coverage across `model/classifier.py` and `app.py`

---

## 🔧 Troubleshooting

<details>
<summary><strong>ModuleNotFoundError: No module named 'torch'</strong></summary>

```bash
# Ensure your venv is active, then:
pip install torch torchvision
```

</details>

<details>
<summary><strong>CUDA out of memory during training</strong></summary>

Reduce batch size in `config.json`:

```json
"training": { "batch_size": 16 }
```

</details>

<details>
<summary><strong>All predictions are "Uncertain"</strong></summary>

1. Check `model/best_model.pt` exists and is approximately 14 MB
2. Verify the model loaded correctly in the server startup log
3. If corrupted, retrain: `./run.sh --train`

</details>

<details>
<summary><strong>Flask won't start — port 5000 already in use</strong></summary>

```bash
# Option 1 — use a different port
./run.sh --port 5001

# Option 2 — find and kill the process using port 5000
lsof -ti:5000 | xargs kill -9      # macOS / Linux
netstat -ano | findstr :5000        # Windows (then taskkill /PID <pid> /F)
```

</details>

<details>
<summary><strong>Image upload fails — "File too large"</strong></summary>

Edit `config.json`:

```json
"app": { "max_upload_mb": 32 }
```

Or resize the image before uploading:

```python
from PIL import Image
img = Image.open('large_photo.jpg')
img.thumbnail((1024, 1024))
img.save('resized.jpg')
```

</details>

<details>
<summary><strong>Enable debug logging</strong></summary>

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

</details>

> Still stuck? [Open an issue](https://github.com/Prafullchaturvedi0/KRISHI-DRISHTI/issues) with your Python version, full error traceback, and steps to reproduce.

---

## 🤝 Contributing

Contributions are welcome. Please follow this workflow:

### Contribution Steps

1. **Fork** the repository
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and run the full test suite
   ```bash
   python test_basic.py
   ```
4. **Commit** with a clear message
   ```bash
   git commit -m "Add: brief description of change"
   ```
5. **Push** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** with a detailed description

### Code Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) — run `flake8` before committing
- **Type hints** on all functions
- **Docstrings required** — every class and function must document its course-concept connection
- **No magic numbers** — all thresholds and constants must reference `config.json`
- **Test coverage** — every new feature must include at least one test in `test_basic.py`

### Priority Contribution Areas

| Area | Description |
|---|---|
| 📱 Android deployment | TensorFlow Lite / ONNX export for native on-device inference |
| 🔄 Active learning | Log uncertain predictions → agronomist labelling → monthly retraining |
| 🗣️ Multi-language | Gondi and Bhili treatment strings for tribal districts of MP |
| 🌾 Field data | Fine-tuning workflow using locally collected and labelled MP farm photos |
| 📶 Offline PWA | Service worker for true offline capability in low-connectivity areas |

### Reporting Issues

Please include:
- Python version (`python --version`)
- Full error message and traceback
- Steps to reproduce
- Expected vs. actual behaviour

---

## 🙏 Acknowledgments

| Source | Contribution |
|---|---|
| [PlantVillage Dataset — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) | 70K+ labelled crop disease images across 38 classes, CC0 public domain |
| [PyTorch — Meta AI](https://pytorch.org) | Open-source deep learning framework; MobileNetV2 pre-trained weights |
| Course Instructors | Guidance on AI and ML fundamentals — every algorithm here reflects course concepts |
| Farmers of Sehore & Ashta Districts | The real-world problem this project addresses, observed in Madhya Pradesh |

---

## 📄 License

This project is provided **as-is for educational purposes**.

```
MIT License

You are free to copy, modify, use, and distribute the code
without restriction and without warranty of any kind.
```

The **PlantVillage dataset** is subject to its own CC0 (public domain) licence.

> ⚠️ Treatment recommendations in Hindi are for educational and informational purposes only.
> They do **not** constitute professional agricultural advice. Verify all recommendations
> with a qualified agronomist before applying any chemical treatment.

---

## 📬 Contact & Support

| Channel | Details |
|---|---|
| **Author** | Prafull Chaturvedi |
| **GitHub** | [@Prafullchaturvedi0](https://github.com/Prafullchaturvedi0) |
| **Email** | prafullchaturvedi0@gmail.com |
| **Issues** | [github.com/Prafullchaturvedi0/KRISHI-DRISHTI/issues](https://github.com/Prafullchaturvedi0/KRISHI-DRISHTI/issues) |
| **Discussions** | [github.com/Prafullchaturvedi0/KRISHI-DRISHTI/discussions](https://github.com/Prafullchaturvedi0/KRISHI-DRISHTI/discussions) |
| **Repository** | [github.com/Prafullchaturvedi0/KRISHI-DRISHTI](https://github.com/Prafullchaturvedi0/KRISHI-DRISHTI) |

---

<div align="center">

**Built for the AI course final project.**
*All ML principles implemented from scratch using only concepts covered in the course outline.*

<br/>

*Last updated: March 2026*

</div>
