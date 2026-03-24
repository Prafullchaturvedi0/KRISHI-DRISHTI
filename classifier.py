"""
KrishiDrishti - Crop Disease Classifier
========================================
Course principles applied:
  - Supervised Learning       : labelled (image, disease) pairs → learn f: X → Y
  - Transfer Learning         : MobileNetV2 pre-trained backbone, fine-tune head
  - Classification            : softmax output over disease classes
  - Feature Learning          : CNN layers learn hierarchical representations
  - Overfitting control       : dropout, data augmentation, early stopping, L2 reg
  - Bias-Variance tradeoff    : validation loss curves guide model complexity choice
  - Probability / Bayesian    : softmax gives P(disease|image); uncertainty flagging
  - Hyperparameter tuning     : learning rate, dropout rate, fine-tune layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import numpy as np
import os
import json
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE CLASSES  (38 PlantVillage classes + local crops)
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Wheat___Brown_rust",
    "Wheat___Yellow_rust",
    "Wheat___Powdery_mildew",
    "Wheat___Healthy",
    "Soybean___Bacterial_pustule",
    "Soybean___Frogeye_leaf_spot",
    "Soybean___Sudden_death_syndrome",
    "Soybean___healthy",
    "Cotton___Leaf_curl_virus",
    "Cotton___Bacterial_blight",
    "Cotton___Alternaria_leaf_spot",
    "Cotton___healthy",
    "Unknown___unclassified",
]

NUM_CLASSES = len(DISEASE_CLASSES)

# Hindi translations for UI output
HINDI_NAMES = {
    "Wheat___Brown_rust":             "गेहूं - भूरी फफूंद",
    "Wheat___Yellow_rust":            "गेहूं - पीली फफूंद",
    "Wheat___Powdery_mildew":         "गेहूं - चूर्णिल आसिता",
    "Wheat___Healthy":                "गेहूं - स्वस्थ",
    "Soybean___Bacterial_pustule":    "सोयाबीन - जीवाणु पस्च्यूल",
    "Soybean___Frogeye_leaf_spot":    "सोयाबीन - मेंढक नेत्र धब्बा",
    "Soybean___Sudden_death_syndrome":"सोयाबीन - अचानक मृत्यु",
    "Soybean___healthy":              "सोयाबीन - स्वस्थ",
    "Cotton___Leaf_curl_virus":       "कपास - पत्ती मुड़न वायरस",
    "Cotton___Bacterial_blight":      "कपास - जीवाणु झुलसा",
    "Cotton___Alternaria_leaf_spot":  "कपास - आल्टरनेरिया धब्बा",
    "Cotton___healthy":               "कपास - स्वस्थ",
    "Tomato___Early_blight":          "टमाटर - अगेती झुलसा",
    "Tomato___Late_blight":           "टमाटर - पछेती झुलसा",
    "Tomato___healthy":               "टमाटर - स्वस्थ",
}

# Treatment recommendations (Hindi)
TREATMENTS = {
    "Wheat___Brown_rust":
        "Propiconazole 25% EC का 0.1% घोल छिड़कें। 15 दिन बाद दोहराएं।",
    "Wheat___Yellow_rust":
        "Tebuconazole 250 EC @ 1ml/L पानी का छिड़काव करें।",
    "Wheat___Powdery_mildew":
        "Sulphur 80% WP @ 2g/L या Triadimefon 25% WP @ 1g/L का छिड़काव करें।",
    "Soybean___Bacterial_pustule":
        "Copper oxychloride 50% WP @ 3g/L पानी का छिड़काव करें।",
    "Soybean___Frogeye_leaf_spot":
        "Carbendazim 50% WP @ 1g/L या Thiophanate methyl का उपयोग करें।",
    "Cotton___Leaf_curl_virus":
        "संक्रमित पौधे उखाड़ें। सफेद मक्खी नियंत्रण हेतु Imidacloprid 17.8 SL @ 0.5ml/L।",
    "Cotton___Bacterial_blight":
        "Streptomycin sulphate 90% + Tetracycline 10% @ 300ppm का छिड़काव करें।",
    "Tomato___Early_blight":
        "Mancozeb 75% WP @ 2.5g/L पानी का छिड़काव 7-10 दिन के अंतर पर।",
    "Tomato___Late_blight":
        "Metalaxyl + Mancozeb @ 2.5g/L का तुरंत छिड़काव करें।",
    "Unknown___unclassified":
        "नमूना निकटतम KVK (कृषि विज्ञान केंद्र) को दिखाएं।",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA AUGMENTATION  (reduces overfitting - course: overfitting & regularization)
# ─────────────────────────────────────────────────────────────────────────────
def get_transforms(mode="train"):
    """
    Training augmentation: flips, rotations, color jitter → expands effective
    dataset size, reducing variance (overfitting) without adding more labelled data.

    Validation/test: only resize + normalize → clean evaluation.
    Normalization uses ImageNet mean/std because we use a pre-trained backbone
    (transfer learning) — the backbone expects this specific input distribution.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),                          # random crop
            transforms.RandomHorizontalFlip(p=0.5),              # flip augment
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),               # rotation augment
            transforms.ColorJitter(                              # lighting variation
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.05
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:  # "val" or "test"
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER LEARNING MODEL  (course: Transfer Learning case study)
# ─────────────────────────────────────────────────────────────────────────────
class CropDiseaseClassifier(nn.Module):
    """
    Transfer Learning Architecture:

    1. Backbone (frozen early layers): MobileNetV2 pre-trained on ImageNet.
       Early conv layers already learned universal features (edges, textures,
       shapes) from 1.2M images — we reuse this knowledge.

    2. Fine-tuned layers: Last 3 blocks of MobileNetV2 are unfrozen and
       trained on crop disease images. These layers adapt to disease-specific
       visual patterns (lesion shapes, color spots, mildew texture).

    3. Custom classification head: Replaces original 1000-class ImageNet head.
       Global Average Pooling → BatchNorm → Dropout → FC → Softmax (38 classes).

    Dropout rate controls bias-variance tradeoff:
      - High dropout → high bias (underfitting), low variance
      - Low dropout  → low bias, high variance (overfitting risk)
      - 0.4 is a well-validated default for fine-tuned CNNs
    """
    def __init__(self, num_classes=NUM_CLASSES, dropout_rate=0.4,
                 fine_tune_blocks=3):
        super().__init__()

        # ── Load pre-trained backbone ──────────────────────────────────────
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )

        # ── Freeze ALL layers first (pure feature extraction) ──────────────
        for param in backbone.parameters():
            param.requires_grad = False

        # ── Unfreeze last N blocks (fine-tuning) ───────────────────────────
        # MobileNetV2 has 19 feature blocks (0–18)
        total_blocks = len(backbone.features)
        unfreeze_from = max(0, total_blocks - fine_tune_blocks)
        for i in range(unfreeze_from, total_blocks):
            for param in backbone.features[i].parameters():
                param.requires_grad = True

        self.features = backbone.features     # feature extractor
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        # ── Custom classification head ─────────────────────────────────────
        in_features = backbone.last_channel   # 1280 for MobileNetV2
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),      # stabilize activations
            nn.Dropout(p=dropout_rate),       # regularization (anti-overfitting)
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),   # lighter dropout in later layer
            nn.Linear(512, num_classes),
            # NOTE: No softmax here — nn.CrossEntropyLoss applies log-softmax
            # internally. For inference, apply softmax manually to get P(class|image)
        )

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """
        Forward pass:
          x: [batch, 3, 224, 224] normalized leaf image tensor
          returns: [batch, num_classes] raw logits
        """
        x = self.features(x)           # CNN feature extraction
        x = self.pool(x)               # global average pooling
        x = torch.flatten(x, 1)        # flatten to [batch, 1280]
        x = self.classifier(x)         # classification head → logits
        return x

    def predict_proba(self, x):
        """
        Returns Bayesian-style posterior P(disease | image) for each class.
        Softmax converts raw logits to a proper probability distribution:
          P(class_k | x) = exp(logit_k) / sum_j(exp(logit_j))

        Course connection: Probability for ML — this IS the conditional
        distribution P(Y=k | X=x) that the model estimates.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENGINE  (course: supervised learning, bias-variance, hyperparameters)
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    """
    Supervised learning training loop.

    Implements:
      - Mini-batch stochastic gradient descent (SGD with momentum)
      - Learning rate scheduling (ReduceLROnPlateau — informed hyperparameter)
      - Early stopping (prevents overfitting when val loss stops improving)
      - Loss & accuracy tracking (bias-variance diagnosis via train vs val curves)
      - Class-weighted loss (handles class imbalance in disease dataset)
    """
    def __init__(self, model, device, class_weights=None):
        self.model = model.to(device)
        self.device = device

        # Loss function: cross-entropy (standard for multi-class classification)
        # With class_weights: upweights rare diseases (e.g. mosaic virus)
        # This is the supervised learning objective: minimize E[L(y, f(x))]
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer: SGD with momentum (convex optimization from course)
        # L2 weight_decay is L2 regularization — penalizes large weights,
        # reduces variance (overfitting), slides along bias-variance tradeoff
        self.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,         # L2 regularization strength
            nesterov=True
        )

        # LR scheduler: halves LR when val loss plateaus (hyperparameter tuning)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [],  "val_acc": [],
            "lr": []
        }

    def train_epoch(self, loader):
        """One full pass over training data."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Backpropagation: compute ∂L/∂w for all trainable parameters
            loss.backward()
            # Gradient clipping: prevents exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on validation or test set (no gradient computation)."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, epochs=30,
            early_stop_patience=7, save_path="best_model.pt"):
        """
        Full training loop with early stopping.

        Early stopping: if validation loss doesn't improve for `patience`
        epochs, stop training. This prevents overfitting — the model would
        otherwise memorize training data and lose generalization.

        Bias-Variance diagnosis:
          - train_loss >> val_loss  → high bias (underfitting)
          - val_loss >> train_loss  → high variance (overfitting)
          - both decrease together → good fit
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        print(f"\n{'='*60}")
        print(f"  KrishiDrishti Training  |  {epochs} epochs max")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc     = self.evaluate(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history for bias-variance curve plotting
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - t0
            gap = val_loss - train_loss  # >0.2 suggests overfitting
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
                f"Gap: {gap:+.4f} | LR: {current_lr:.6f} | {elapsed:.1f}s"
            )

            # LR scheduling based on validation loss
            self.scheduler.step(val_loss)

            # Early stopping + best model checkpoint
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "history": self.history,
                    "classes": DISEASE_CLASSES,
                }, save_path)
                print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{early_stop_patience}")
                if patience_counter >= early_stop_patience:
                    print(f"\n  Early stopping at epoch {epoch}. "
                          f"Best epoch was {best_epoch}.")
                    break

        print(f"\n{'='*60}")
        print(f"  Training complete. Best val_loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")
        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER  (80/10/10 train/val/test split)
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(data_dir, batch_size=32, val_frac=0.1, test_frac=0.1,
                      num_workers=4):
    """
    Builds train/val/test DataLoaders from an ImageFolder directory.

    Split strategy (course: hyperparameters and validation sets):
      - 80% train   : model sees these examples
      - 10% val     : tune hyperparameters, detect overfitting
      - 10% test    : held-out evaluation; touched ONLY at the end

    The test set must never be used during hyperparameter search — it is
    the unbiased estimator of true generalization performance.
    """
    # Load full dataset with training augmentation (for split first)
    full_dataset = ImageFolder(data_dir, transform=get_transforms("train"))
    n = len(full_dataset)

    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)   # reproducibility
    )

    # Validation & test use clean transforms (no augmentation)
    val_ds.dataset.transform  = get_transforms("val")
    test_ds.dataset.transform = get_transforms("val")

    print(f"Dataset splits — Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, full_dataset.classes


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE  (course: Bayesian statistics, probability, classification)
# ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine:
    """
    Single-image inference with Bayesian uncertainty quantification.

    The model outputs P(disease_k | image) for all k.
    We use this posterior distribution to:
      1. Identify the MAP (maximum a posteriori) class → top prediction
      2. Flag uncertain predictions (entropy-based) → "consult expert"
      3. Return top-3 predictions → Bayesian reasoning from syllabus
    """
    UNCERTAINTY_THRESHOLD = 0.60   # below this confidence → flag as uncertain
    ENTROPY_THRESHOLD     = 2.5    # high entropy → model is confused

    def __init__(self, model_path, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.transform = get_transforms("val")
        self._load_model(model_path)

    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = CropDiseaseClassifier(num_classes=NUM_CLASSES)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.classes = checkpoint.get("classes", DISEASE_CLASSES)
        print(f"Model loaded from {model_path} "
              f"(val_acc={checkpoint.get('val_acc', '?'):.3f})")

    def predict(self, pil_image):
        """
        Full inference pipeline for one PIL image.

        Returns a dict with:
          - top_class      : predicted disease name
          - confidence     : P(top_class | image) — the posterior probability
          - top3           : [(class, prob), ...] top 3 predictions
          - uncertain      : True if model lacks confidence (consult expert)
          - entropy        : Shannon entropy H(P) — measures prediction spread
          - hindi_name     : local language class name
          - treatment      : recommended treatment in Hindi
        """
        # Preprocess
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Forward pass → probabilities
        probs = self.model.predict_proba(tensor)[0].cpu().numpy()  # [NUM_CLASSES]

        # MAP prediction (argmax of posterior)
        top_idx = int(np.argmax(probs))
        top_class = self.classes[top_idx]
        confidence = float(probs[top_idx])

        # Top-3 predictions (Bayesian reasoning: show alternative hypotheses)
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {"class": self.classes[i],
             "probability": round(float(probs[i]), 4),
             "hindi": HINDI_NAMES.get(self.classes[i], self.classes[i])}
            for i in top3_indices
        ]

        # Shannon entropy: H = -sum(p * log(p))
        # Low entropy  → model is confident (peaked distribution)
        # High entropy → model is uncertain (flat distribution)
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        uncertain = (confidence < self.UNCERTAINTY_THRESHOLD or
                     entropy > self.ENTROPY_THRESHOLD)

        # Determine disease category
        is_healthy = "healthy" in top_class.lower()
        treatment = (
            "पौधा स्वस्थ दिखता है। नियमित देखभाल जारी रखें।"
            if is_healthy
            else TREATMENTS.get(top_class, TREATMENTS["Unknown___unclassified"])
        )

        return {
            "top_class":   top_class,
            "confidence":  round(confidence, 4),
            "top3":        top3,
            "uncertain":   uncertain,
            "entropy":     round(entropy, 4),
            "is_healthy":  is_healthy,
            "hindi_name":  HINDI_NAMES.get(top_class, top_class.replace("_", " ")),
            "treatment":   treatment,
            "probabilities": {
                self.classes[i]: round(float(probs[i]), 6)
                for i in np.argsort(probs)[::-1][:10]
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION  (course: estimators, bias, variance, confusion matrix)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, test_loader, class_names, device):
    """
    Full evaluation on held-out test set.
    Computes:
      - Overall accuracy (unbiased estimator of generalization)
      - Per-class accuracy (identifies biased classes)
      - Macro F1 (handles class imbalance — from statistical decision theory)
      - Confusion matrix (reveals bias toward common classes)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall accuracy
    accuracy = float(np.mean(all_preds == all_labels))

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_acc[name] = float(np.mean(all_preds[mask] == i))

    # Macro F1 (manual implementation — course principle, not sklearn)
    f1_scores = []
    for i in range(len(class_names)):
        tp = np.sum((all_preds == i) & (all_labels == i))
        fp = np.sum((all_preds == i) & (all_labels != i))
        fn = np.sum((all_preds != i) & (all_labels == i))
        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores))

    results = {
        "accuracy":      round(accuracy, 4),
        "macro_f1":      round(macro_f1, 4),
        "per_class_acc": {k: round(v, 4) for k, v in per_class_acc.items()},
        "num_samples":   len(all_labels),
    }

    print(f"\nTest Set Evaluation")
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"  Samples   : {len(all_labels)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def train(data_dir="data/plantvillage", epochs=30, batch_size=32,
          dropout_rate=0.4, fine_tune_blocks=3, save_path="model/best_model.pt"):
    """
    Complete training pipeline.
    Adjust hyperparameters here to navigate bias-variance tradeoff:
      dropout_rate=0.2, fine_tune_blocks=5 → lower bias, higher variance
      dropout_rate=0.6, fine_tune_blocks=1 → higher bias, lower variance
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir, batch_size=batch_size
    )

    model = CropDiseaseClassifier(
        num_classes=len(class_names),
        dropout_rate=dropout_rate,
        fine_tune_blocks=fine_tune_blocks
    )

    trainer = Trainer(model, device)
    history = trainer.fit(train_loader, val_loader,
                          epochs=epochs, save_path=save_path)

    # Final evaluation on held-out test set
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    results = evaluate_model(model, test_loader, class_names, device)

    # Save history for plotting
    with open("model/training_history.json", "w") as f:
        json.dump({"history": history, "evaluation": results}, f, indent=2)

    return history, results


if __name__ == "__main__":
    train()
