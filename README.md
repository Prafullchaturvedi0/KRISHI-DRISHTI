# KrishiDrishti — कृषि दृष्टि
## AI-Powered Crop Disease Detection | Fundamentals In AI and ML Course Project

A complete, runnable implementation of the KrishiDrishti project.
Every design decision maps directly to a topic in the course outline.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web app (DEMO mode — no trained model needed)
python app.py

# 3. Open in browser
http://localhost:5000
```

The system runs in **DEMO mode** out of the box — it returns realistic simulated
predictions so you can see the full UI and probability outputs immediately,
without needing a trained model or GPU.

---

## To Train the Real Model

```bash
# 1. Download PlantVillage dataset
#    https://www.kaggle.com/datasets/emmarex/plantdisease
#    Extract to: data/plantvillage/

# 2. Train
python -c "from model.classifier import train; train()"

# 3. Model saved to model/best_model.pt
#    Training history saved to model/training_history.json
#    Restart the server — it auto-loads the trained model
```

---

## Project Structure

```
krishidrishti/
├── model/
│   └── classifier.py       # All ML code (see course mapping below)
├── templates/
│   └── index.html          # Full HTML/JS frontend interface
├── app.py                  # Flask server (connects model ↔ browser)
├── requirements.txt
└── README.md
```

---

## Course Concept → Code Mapping

| Course Topic                    | Where in Code                              | What It Does                                    |
|----------------------------------|--------------------------------------------|-------------------------------------------------|
| Intelligent Agents               | `InferenceEngine` class                    | PEAS agent: perceives image, acts with diagnosis |
| Search (Hyperparameter tuning)   | `ReduceLROnPlateau`, `Trainer.fit()`       | Adaptive informed search over LR schedule       |
| Probability Theory               | `predict_proba()`, softmax                 | P(disease\|image) — conditional distribution    |
| Convex Optimization              | SGD + momentum + weight_decay              | Minimizes cross-entropy loss                    |
| Statistical Decision Theory      | Class-weighted cross-entropy               | Prior P(disease) from seasonal prevalence       |
| Supervised Learning              | `build_dataloaders()`, `Trainer`           | Labelled (image, label) → learn f: X→Y          |
| Classification                   | `CropDiseaseClassifier`, softmax head      | 38-class multi-class classification             |
| Overfitting & Regularization     | Dropout, augmentation, early stopping, L2  | Four complementary anti-overfitting methods     |
| Hyperparameters & Validation     | 80/10/10 split, val loss monitoring        | Tune on val set, evaluate once on test set      |
| Estimators, Bias & Variance      | Loss curves, train vs val gap              | Bias-variance diagnostic visualization         |
| Bayesian Statistics              | Entropy thresholding, uncertain flag       | Posterior uncertainty → expert referral         |
| Curse of Dimensionality          | Conv layers, global avg pool               | Project 150K-d image → 1280-d manifold          |
| Feature Learning                 | Frozen MobileNetV2 blocks                  | Hierarchical: edges → textures → lesions        |
| Transfer Learning                | Pre-trained backbone + fine-tuned head     | ImageNet knowledge → crop disease domain        |
| NLP / Sentiment (adjacent)       | Hindi output, treatment recommendations    | Language-localised actionable output            |

---

## Architecture Decisions Explained

### Why MobileNetV2 (not ResNet-50)?
- Size: 14 MB vs 98 MB — fits on low-end Android phones
- Inference: ~0.8s on CPU vs ~3s for ResNet-50
- Accuracy: within 2% of ResNet-50 on PlantVillage
- **Bias-variance**: lighter model = lower variance risk with small local dataset

### Why freeze early layers?
- Early CNN layers learn universal features (edges, textures) — same across ImageNet and leaf images
- Fine-tuning them requires more data and risks catastrophic forgetting
- Only the last 3 blocks adapt to disease-specific visual patterns

### Why dropout = 0.4?
- Empirically optimal for fine-tuned CNNs on mid-size datasets
- Too high (0.7): underfits (high bias) — can't learn disease patterns
- Too low (0.1): overfits (high variance) — memorizes training images

### Why Macro F1 over accuracy?
- Rust is 10× more common than mosaic virus in PlantVillage
- Accuracy rewards always predicting "rust" → misleadingly high score
- Macro F1 gives equal weight to each disease class — aligns with real-world impact

---

## Demo Mode

When no `model/best_model.pt` exists, the server returns realistic simulated
predictions from four representative scenarios:
1. High confidence disease (wheat brown rust, 87%)
2. Viral disease (cotton leaf curl, 76%)
3. Healthy plant (tomato healthy, 94%)
4. Uncertain prediction (soybean frogeye, 52% → expert referral triggered)

---

*Built for the AI course final project. All ML principles implemented from scratch
using only concepts covered in the course outline.*
