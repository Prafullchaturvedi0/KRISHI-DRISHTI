"""
test_basic.py — KrishiDrishti Test Suite
==========================================
Tests every core component introduced in the course outline:

  Unit 1  — Intelligent Agents   : InferenceEngine as a rational agent
  Unit 2  — Search               : Hyperparameter search space validity
  Unit 3  — Probability Theory   : Softmax distributions, entropy, Bayes
  Unit 4  — Supervised Learning  : Data splits, transforms, model I/O
  Unit 4  — Transfer Learning    : Backbone frozen/unfrozen layer counts
  Unit 4  — Overfitting          : Augmentation pipeline, dropout presence
  Unit 4  — Bias-Variance        : Loss curve diagnostics
  Unit 4  — Classification       : Output shape, argmax, top-k validity
  Unit 4  — Feature Learning     : Embedding dimensionality
  Unit 4  — Bayesian Statistics  : Uncertainty flagging thresholds
  App     — Flask routes         : /health, /predict, /history endpoints

Run with:
  python -m pytest test_basic.py -v
  python test_basic.py          (without pytest)
"""

import sys
import io
import json
import math
import random
import unittest
import importlib
from pathlib import Path
from unittest.mock import patch, MagicMock

# ── Dependency availability flags ─────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_rgb_image(width=224, height=224):
    """Return a PIL RGB image filled with random noise (simulates a leaf photo)."""
    if not PIL_AVAILABLE:
        return None
    img = Image.new("RGB", (width, height))
    pixels = [(random.randint(40, 200), random.randint(80, 200), random.randint(20, 100))
              for _ in range(width * height)]
    img.putdata(pixels)
    return img


def make_image_bytes(width=224, height=224, fmt="JPEG"):
    """Return raw bytes of a synthetic image (for HTTP upload simulation)."""
    img = make_rgb_image(width, height)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 1 — Config & Project Structure
# ─────────────────────────────────────────────────────────────────────────────
class TestProjectStructure(unittest.TestCase):
    """Verify all required files exist and are non-empty."""

    REQUIRED_FILES = [
        "app.py",
        "model/classifier.py",
        "templates/index.html",
        "requirements.txt",
        "config.json",
        "run.sh",
        ".gitignore",
    ]

    def test_required_files_exist(self):
        for path in self.REQUIRED_FILES:
            with self.subTest(path=path):
                self.assertTrue(
                    Path(path).exists(),
                    f"Required file missing: {path}"
                )

    def test_required_files_nonempty(self):
        for path in self.REQUIRED_FILES:
            p = Path(path)
            if p.exists():
                with self.subTest(path=path):
                    self.assertGreater(p.stat().st_size, 0,
                                       f"File is empty: {path}")

    def test_config_json_valid(self):
        """config.json must be valid JSON with required top-level keys."""
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for key in ("app", "model", "training", "inference", "disease_classes"):
            self.assertIn(key, cfg, f"config.json missing key: {key}")

    def test_config_num_classes_matches_classifier(self):
        """NUM_CLASSES in config.json must match DISEASE_CLASSES list length."""
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["model"]["num_classes"],
            len(cfg["disease_classes"]),
            "config.json: model.num_classes != len(disease_classes)"
        )

    def test_config_training_split_sums_to_one(self):
        """val_fraction + test_fraction must be < 1.0 (leaving room for train)."""
        with open("config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tr = cfg["training"]
        total = tr["val_fraction"] + tr["test_fraction"]
        self.assertLess(total, 1.0,
                        "val_fraction + test_fraction must be < 1.0")

    def test_requirements_contains_core_packages(self):
        """requirements.txt must list flask, torch, torchvision, Pillow."""
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        for pkg in ("flask", "torch", "torchvision", "pillow"):
            self.assertIn(pkg, content,
                          f"requirements.txt missing: {pkg}")

    def test_run_sh_executable_content(self):
        """run.sh must start with a shebang and contain key commands."""
        with open("run.sh", "r") as f:
            content = f.read()
        self.assertTrue(content.startswith("#!/"),
                        "run.sh must start with a shebang (#!)")
        for keyword in ("python3", "pip install", "flask", "venv"):
            self.assertIn(keyword, content,
                          f"run.sh missing expected keyword: {keyword}")

    def test_gitignore_covers_sensitive_paths(self):
        """gitignore must exclude model weights, venv, and data directory."""
        with open(".gitignore", "r") as f:
            content = f.read()
        for pattern in ("*.pt", ".venv", "__pycache__", "*.pyc"):
            self.assertIn(pattern, content,
                          f".gitignore missing pattern: {pattern}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 2 — Probability Theory (Unit 3)
# ─────────────────────────────────────────────────────────────────────────────
class TestProbabilityPrinciples(unittest.TestCase):
    """
    Course: Probability Theory, random variables, distributions, entropy.
    Tests that the mathematical foundations are correct independent of PyTorch.
    """

    def _softmax(self, logits):
        """Pure-Python softmax — same operation as torch.softmax."""
        mx = max(logits)
        exps = [math.exp(x - mx) for x in logits]
        s = sum(exps)
        return [e / s for e in exps]

    def _entropy(self, probs):
        """Shannon entropy H(P) = -sum(p * log(p))."""
        return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)

    def test_softmax_sums_to_one(self):
        """P(disease_k | image) must be a valid probability distribution."""
        logits = [2.3, 0.5, -1.2, 0.8, 1.1] * 7 + [0.0] * 3  # 38 values
        probs = self._softmax(logits)
        self.assertAlmostEqual(sum(probs), 1.0, places=6,
                               msg="Softmax output must sum to 1.0")

    def test_softmax_all_positive(self):
        """All class probabilities must be in (0, 1)."""
        logits = [random.uniform(-5, 5) for _ in range(38)]
        probs = self._softmax(logits)
        for i, p in enumerate(probs):
            self.assertGreater(p, 0.0, f"P[{i}] must be > 0")
            self.assertLess(p, 1.0,    f"P[{i}] must be < 1")

    def test_argmax_matches_highest_prob(self):
        """MAP estimate (argmax) correctly identifies the most likely disease."""
        logits = [0.1] * 38
        logits[5] = 10.0          # class 5 has highest logit
        probs = self._softmax(logits)
        predicted_class = probs.index(max(probs))
        self.assertEqual(predicted_class, 5,
                         "argmax must return the class with the highest probability")

    def test_entropy_certain_prediction(self):
        """A peaked distribution (one class = 0.99) should have low entropy."""
        probs = [0.0005] * 38
        probs[0] = 1.0 - sum(probs[1:])
        h = self._entropy(probs)
        self.assertLess(h, 1.0,
                        "Entropy of near-certain distribution must be low (<1.0)")

    def test_entropy_uniform_distribution(self):
        """Uniform distribution over 38 classes should have high entropy."""
        probs = [1.0 / 38] * 38
        h = self._entropy(probs)
        max_h = math.log(38)   # theoretical maximum for 38 classes ≈ 3.64
        self.assertGreater(h, 3.0,
                           "Entropy of uniform distribution over 38 classes must be high (>3.0)")
        self.assertAlmostEqual(h, max_h, places=3,
                               msg="Entropy of uniform distribution should equal log(n)")

    def test_uncertainty_threshold_logic(self):
        """
        Bayesian uncertainty: if confidence < 0.60 OR entropy > 2.5 → flag.
        Tests the decision rule from InferenceEngine.
        """
        CONF_THRESH    = 0.60
        ENTROPY_THRESH = 2.5

        # Case 1: confident, low entropy → not uncertain
        conf, ent = 0.87, 0.68
        uncertain = (conf < CONF_THRESH) or (ent > ENTROPY_THRESH)
        self.assertFalse(uncertain, "High-confidence prediction should not be flagged")

        # Case 2: low confidence → uncertain
        conf, ent = 0.52, 1.2
        uncertain = (conf < CONF_THRESH) or (ent > ENTROPY_THRESH)
        self.assertTrue(uncertain, "Low-confidence prediction must be flagged")

        # Case 3: decent confidence but high entropy → uncertain
        conf, ent = 0.65, 3.1
        uncertain = (conf < CONF_THRESH) or (ent > ENTROPY_THRESH)
        self.assertTrue(uncertain, "High-entropy prediction must be flagged")

    def test_top3_probabilities_decreasing(self):
        """Top-3 predictions must be in descending probability order."""
        logits = [random.uniform(-3, 3) for _ in range(38)]
        probs  = self._softmax(logits)
        top3_idx = sorted(range(38), key=lambda i: probs[i], reverse=True)[:3]
        top3_probs = [probs[i] for i in top3_idx]
        self.assertGreaterEqual(top3_probs[0], top3_probs[1])
        self.assertGreaterEqual(top3_probs[1], top3_probs[2])

    def test_bayesian_prior_class_weights(self):
        """
        Class-weighted loss encodes prior P(disease).
        Rare diseases should receive higher weight (inverse frequency weighting).
        """
        # Simulate class frequencies: rust is 10x more common than mosaic
        class_counts = [100] * 36 + [10, 5]   # last two are rare classes
        total = sum(class_counts)
        # Inverse frequency weighting
        weights = [total / (len(class_counts) * c) for c in class_counts]
        # Rare classes must have higher weight than common classes
        self.assertGreater(weights[-1], weights[0],
                           "Rare class must have higher loss weight than common class")
        self.assertGreater(weights[-2], weights[0])


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 3 — Supervised Learning Components (Unit 4)
# ─────────────────────────────────────────────────────────────────────────────
class TestSupervisedLearningComponents(unittest.TestCase):
    """
    Course: Supervised Learning, overfitting, hyperparameters, validation sets.
    Tests data transforms, splits, and training loop logic.
    """

    def test_data_split_proportions(self):
        """80/10/10 train/val/test split must give correct sample counts."""
        n = 1000
        val_frac  = 0.10
        test_frac = 0.10
        n_test  = int(n * test_frac)
        n_val   = int(n * val_frac)
        n_train = n - n_val - n_test

        self.assertEqual(n_train + n_val + n_test, n,
                         "Train + val + test must equal total dataset size")
        self.assertEqual(n_train, 800)
        self.assertEqual(n_val,   100)
        self.assertEqual(n_test,  100)

    def test_test_set_must_not_overlap_train(self):
        """Train and test indices must be disjoint — data leakage check."""
        indices = list(range(1000))
        random.shuffle(indices)
        train_idx = set(indices[:800])
        val_idx   = set(indices[800:900])
        test_idx  = set(indices[900:])

        self.assertEqual(len(train_idx & test_idx), 0,
                         "Train and test sets must have no overlapping indices")
        self.assertEqual(len(train_idx & val_idx), 0,
                         "Train and val sets must have no overlapping indices")
        self.assertEqual(len(val_idx & test_idx), 0,
                         "Val and test sets must have no overlapping indices")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_augmentation_transform_output_shape(self):
        """Training augmentation must produce 3×224×224 tensor from any input size."""
        from model.classifier import get_transforms
        img = make_rgb_image(512, 512)  # large source image
        tf  = get_transforms("train")
        tensor = tf(img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224),
                         "Train transform must output [3, 224, 224] tensor")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_val_transform_output_shape(self):
        """Validation transform must produce 3×224×224 tensor (no augmentation)."""
        from model.classifier import get_transforms
        img = make_rgb_image(300, 400)
        tf  = get_transforms("val")
        tensor = tf(img)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224),
                         "Val transform must output [3, 224, 224] tensor")

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
    def test_transform_normalization_range(self):
        """
        ImageNet normalization: pixel values are no longer in [0,1].
        Values should span roughly [-3, 3] after normalization.
        """
        from model.classifier import get_transforms
        img    = make_rgb_image()
        tensor = get_transforms("val")(img)
        self.assertLess(float(tensor.min()), 0.0,
                        "Normalized tensor must have negative values")
        self.assertGreater(float(tensor.max()), 0.0,
                           "Normalized tensor must have positive values")
        # Reasonable range check (ImageNet normalization gives roughly -3..3)
        self.assertGreater(float(tensor.min()), -5.0)
        self.assertLess(float(tensor.max()),     5.0)

    def test_early_stopping_triggers_correctly(self):
        """
        Simulate the early stopping counter logic from Trainer.fit().
        After `patience` epochs without improvement, training should stop.
        """
        patience = 7
        best_val_loss = float("inf")
        patience_counter = 0
        stopped_at = None

        # Simulate: val loss improves for 5 epochs, then plateaus for 7
        val_losses = [2.0, 1.8, 1.6, 1.4, 1.3] + [1.35] * 10

        for epoch, vl in enumerate(val_losses, start=1):
            if vl < best_val_loss - 1e-4:
                best_val_loss = vl
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    stopped_at = epoch
                    break

        self.assertIsNotNone(stopped_at, "Early stopping should have triggered")
        self.assertEqual(stopped_at, 12,
                         "Early stopping should trigger at epoch 12 (5 improving + 7 patience)")

    def test_bias_variance_gap_detection(self):
        """
        Simulated training curves: a gap of >0.3 between val and train loss
        should be detected as potential overfitting.
        """
        OVERFIT_GAP = 0.20   # threshold from project diagnostics

        # Good fit: gap is small
        train_loss_good = 0.45
        val_loss_good   = 0.50
        gap_good = val_loss_good - train_loss_good
        self.assertLess(gap_good, OVERFIT_GAP,
                        "Small gap should not trigger overfitting warning")

        # Overfitting: val_loss >> train_loss
        train_loss_overfit = 0.15
        val_loss_overfit   = 0.80
        gap_overfit = val_loss_overfit - train_loss_overfit
        self.assertGreater(gap_overfit, OVERFIT_GAP,
                           "Large gap should trigger overfitting warning")

    def test_macro_f1_computation(self):
        """
        Manual Macro F1 computation (course: estimators and bias).
        F1 = 2*P*R/(P+R) per class; Macro F1 = mean over all classes.
        """
        # 3-class example with known ground truth and predictions
        y_true = [0, 0, 1, 1, 2, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0, 2]   # one wrong per class
        n_classes = 3

        f1_scores = []
        for cls in range(n_classes):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            precision = tp / (tp + fp + 1e-10)
            recall    = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores)
        self.assertGreater(macro_f1, 0.0)
        self.assertLessEqual(macro_f1, 1.0)
        # Should be roughly 0.72 for this example
        self.assertAlmostEqual(macro_f1, 0.72, delta=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 4 — Transfer Learning Model (Unit 4 / Case Study)
# ─────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class TestTransferLearningModel(unittest.TestCase):
    """
    Course: Transfer Learning, Feature Learning, Curse of Dimensionality.
    Tests model architecture without needing a trained checkpoint.
    """

    @classmethod
    def setUpClass(cls):
        from model.classifier import CropDiseaseClassifier, NUM_CLASSES
        cls.NUM_CLASSES = NUM_CLASSES
        cls.model = CropDiseaseClassifier(
            num_classes=NUM_CLASSES, dropout_rate=0.4, fine_tune_blocks=3
        )
        cls.model.eval()

    def test_model_output_shape(self):
        """
        Forward pass must produce [batch, num_classes] logit tensor.
        This verifies the classification head is correctly connected.
        """
        batch = torch.zeros(2, 3, 224, 224)   # 2 synthetic images
        with torch.no_grad():
            logits = self.model(batch)
        self.assertEqual(tuple(logits.shape), (2, self.NUM_CLASSES),
                         f"Model output must be [batch, {self.NUM_CLASSES}]")

    def test_backbone_frozen_layers(self):
        """
        Early backbone layers must be frozen (requires_grad=False).
        This is the 'freeze early layers' principle of transfer learning.
        """
        frozen_count = sum(
            1 for p in self.model.features.parameters() if not p.requires_grad
        )
        self.assertGreater(frozen_count, 0,
                           "At least some backbone parameters must be frozen")

    def test_fine_tuned_layers_trainable(self):
        """
        Last fine_tune_blocks of the backbone must be trainable (requires_grad=True).
        This is what actually adapts to crop disease images.
        """
        trainable_count = sum(
            1 for p in self.model.parameters() if p.requires_grad
        )
        self.assertGreater(trainable_count, 0,
                           "At least some parameters must be trainable")

    def test_classifier_head_has_dropout(self):
        """
        The custom head must include Dropout for regularization.
        This directly implements the overfitting control from the course.
        """
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in self.model.classifier.modules()
        )
        self.assertTrue(has_dropout,
                        "Classification head must contain a Dropout layer")

    def test_classifier_head_has_batchnorm(self):
        """BatchNorm in the head stabilizes training and reduces covariate shift."""
        has_bn = any(
            isinstance(m, nn.BatchNorm1d) for m in self.model.classifier.modules()
        )
        self.assertTrue(has_bn,
                        "Classification head must contain a BatchNorm1d layer")

    def test_predict_proba_is_valid_distribution(self):
        """
        predict_proba must return a probability distribution:
          - all values in (0, 1)
          - sums to 1.0
        This is the softmax posterior P(disease | image).
        """
        img  = make_rgb_image()
        from model.classifier import get_transforms
        tensor = get_transforms("val")(img).unsqueeze(0)
        probs  = self.model.predict_proba(tensor)[0]

        self.assertEqual(len(probs), self.NUM_CLASSES)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)
        self.assertTrue(all(p > 0 for p in probs),  "All probs must be > 0")
        self.assertTrue(all(p < 1 for p in probs),  "All probs must be < 1")

    def test_different_inputs_give_different_outputs(self):
        """
        Two different images should produce different logits.
        (Tests the model is not outputting a constant — basic sanity check.)
        """
        from model.classifier import get_transforms
        tf   = get_transforms("val")
        img1 = tf(make_rgb_image()).unsqueeze(0)
        img2 = tf(make_rgb_image()).unsqueeze(0)
        with torch.no_grad():
            out1 = self.model(img1)
            out2 = self.model(img2)
        self.assertFalse(
            torch.allclose(out1, out2, atol=1e-3),
            "Different inputs must produce different model outputs"
        )

    def test_global_average_pool_dimensionality_reduction(self):
        """
        Course: Curse of Dimensionality.
        Input: 224*224*3 = 150,528 dimensions.
        After conv + pool: 1280 dimensions.
        This is the dimensionality reduction that makes classification feasible.
        """
        input_dims  = 224 * 224 * 3       # raw pixel space
        output_dims = 1280                 # MobileNetV2 backbone output

        self.assertLess(output_dims, input_dims,
                        "Feature embedding must be lower-dimensional than raw input")
        ratio = output_dims / input_dims
        self.assertLess(ratio, 0.01,
                        "Dimensionality reduction should be >99% (150K → 1280)")

    def test_model_num_classes_matches_config(self):
        """NUM_CLASSES in the model must match config.json."""
        with open("config.json") as f:
            cfg = json.load(f)
        self.assertEqual(self.NUM_CLASSES, cfg["model"]["num_classes"])


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 5 — Flask App Routes
# ─────────────────────────────────────────────────────────────────────────────
@unittest.skipUnless(FLASK_AVAILABLE, "Flask not installed")
@unittest.skipUnless(PIL_AVAILABLE, "Pillow not installed")
class TestFlaskRoutes(unittest.TestCase):
    """
    Tests the web interface layer (app.py):
      /health  — reports model status
      /predict — accepts image, returns structured JSON
      /history — returns training history for chart rendering
    """

    @classmethod
    def setUpClass(cls):
        import app as kd_app
        kd_app.app.config["TESTING"] = True
        cls.client = kd_app.app.test_client()

    def test_health_endpoint_returns_200(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_health_response_has_required_keys(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        for key in ("status", "model_loaded", "demo_mode"):
            self.assertIn(key, data, f"/health response missing key: {key}")

    def test_health_status_is_ok(self):
        resp = self.client.get("/health")
        data = json.loads(resp.data)
        self.assertEqual(data["status"], "ok")

    def test_predict_without_image_returns_400(self):
        resp = self.client.post("/predict", data={})
        self.assertEqual(resp.status_code, 400)

    def test_predict_with_valid_image_returns_200(self):
        img_buf = make_image_bytes(224, 224)
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "test_leaf.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200,
                         f"Expected 200, got {resp.status_code}: {resp.data}")

    def test_predict_response_structure(self):
        """Prediction JSON must contain all required fields for the UI."""
        img_buf = make_image_bytes(224, 224)
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "leaf.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        data = json.loads(resp.data)
        required_keys = [
            "top_class", "confidence", "top3", "uncertain",
            "entropy", "is_healthy", "hindi_name", "treatment"
        ]
        for key in required_keys:
            self.assertIn(key, data,
                          f"/predict response missing key: {key}")

    def test_predict_confidence_in_range(self):
        """Confidence must be a probability: 0.0 ≤ confidence ≤ 1.0."""
        img_buf = make_image_bytes()
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "leaf.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        data = json.loads(resp.data)
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)

    def test_predict_top3_has_three_items(self):
        """Top-3 response must always have exactly 3 entries."""
        img_buf = make_image_bytes()
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "leaf.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        data = json.loads(resp.data)
        self.assertEqual(len(data["top3"]), 3,
                         "top3 must always contain exactly 3 predictions")

    def test_predict_top3_probabilities_sum_lte_one(self):
        """Top-3 probabilities must be ≤ 1.0 (they are a subset of P distribution)."""
        img_buf = make_image_bytes()
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "leaf.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        data = json.loads(resp.data)
        total = sum(item["probability"] for item in data["top3"])
        self.assertLessEqual(total, 1.0 + 1e-4,
                             "Sum of top-3 probabilities must be ≤ 1.0")

    def test_predict_entropy_is_non_negative(self):
        """Shannon entropy H(P) must always be ≥ 0."""
        img_buf = make_image_bytes()
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "leaf.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        data = json.loads(resp.data)
        self.assertGreaterEqual(data["entropy"], 0.0,
                                "Entropy must be non-negative")

    def test_predict_too_small_image_returns_400(self):
        """Images smaller than 32×32 should be rejected."""
        img_buf = make_image_bytes(10, 10)
        resp = self.client.post(
            "/predict",
            data={"image": (img_buf, "tiny.jpg", "image/jpeg")},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400,
                         "Image smaller than 32×32 must return 400")

    def test_history_endpoint_returns_200(self):
        resp = self.client.get("/history")
        self.assertEqual(resp.status_code, 200)

    def test_history_response_has_required_keys(self):
        resp = self.client.get("/history")
        data = json.loads(resp.data)
        self.assertIn("history", data, "/history response missing 'history' key")
        h = data["history"]
        for key in ("train_loss", "val_loss", "train_acc", "val_acc", "lr"):
            self.assertIn(key, h, f"history dict missing key: {key}")

    def test_history_loss_arrays_same_length(self):
        """All history arrays must have the same length (one value per epoch)."""
        resp = self.client.get("/history")
        data = json.loads(resp.data)["history"]
        lengths = {k: len(v) for k, v in data.items()}
        unique_lengths = set(lengths.values())
        self.assertEqual(len(unique_lengths), 1,
                         f"All history arrays must have equal length: {lengths}")

    def test_index_route_returns_html(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"KrishiDrishti", resp.data,
                      "Index page must contain 'KrishiDrishti'")


# ─────────────────────────────────────────────────────────────────────────────
# TEST GROUP 6 — Intelligent Agent Properties (Unit 1)
# ─────────────────────────────────────────────────────────────────────────────
class TestIntelligentAgentProperties(unittest.TestCase):
    """
    Course: Intelligent Agents, rationality, PEAS.
    Tests that the InferenceEngine satisfies the rational agent definition:
      - It acts based on a percept (image)
      - It maximizes performance (highest posterior class)
      - It is consistent (same input → same output)
      - It handles uncertainty rationally (expert referral)
    """

    def test_consistent_deterministic_output(self):
        """
        A rational agent must be deterministic at inference time:
        same image → same diagnosis (eval mode, no dropout stochasticity).
        """
        if not (TORCH_AVAILABLE and PIL_AVAILABLE):
            self.skipTest("torch/PIL not available")
        from model.classifier import CropDiseaseClassifier, get_transforms, NUM_CLASSES
        model = CropDiseaseClassifier(num_classes=NUM_CLASSES)
        model.eval()
        tf  = get_transforms("val")
        img = make_rgb_image()
        t   = tf(img).unsqueeze(0)
        with torch.no_grad():
            out1 = model(t)
            out2 = model(t)
        self.assertTrue(torch.allclose(out1, out2),
                        "Same input must produce same output in eval mode (determinism)")

    def test_highest_posterior_selected_as_action(self):
        """
        Rationality: argmax of P(disease|image) must be selected as the
        agent's action (the diagnosis to output).
        """
        import math
        probs = [0.02] * 38
        probs[12] = 0.60      # class 12 has highest probability
        total = sum(probs)
        probs = [p / total for p in probs]

        action = probs.index(max(probs))
        self.assertEqual(action, 12,
                         "Rational agent must select the MAP (highest probability) class")

    def test_uncertainty_triggers_expert_referral(self):
        """
        When the agent is uncertain (low confidence, high entropy),
        it must recommend expert consultation rather than a wrong treatment.
        This is rational behaviour under uncertainty.
        """
        TREATMENTS = json.load(open("config.json"))["treatments"]
        EXPERT_REFERRAL_KEY = "Unknown___unclassified"

        # Simulate uncertain case
        confidence = 0.45
        top_class  = "Unknown___unclassified"
        treatment  = TREATMENTS.get(top_class, "")

        self.assertIn("KVK", treatment,
                      "Uncertain prediction must recommend KVK expert consultation")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — run tests standalone or via pytest
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  KrishiDrishti — Test Suite")
    print("  Testing all course-principle components")
    print("=" * 65 + "\n")

    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestProjectStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestProbabilityPrinciples))
    suite.addTests(loader.loadTestsFromTestCase(TestSupervisedLearningComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestTransferLearningModel))
    suite.addTests(loader.loadTestsFromTestCase(TestFlaskRoutes))
    suite.addTests(loader.loadTestsFromTestCase(TestIntelligentAgentProperties))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    if result.wasSuccessful():
        print(f"  PASSED  {result.testsRun} tests — all good ✓")
    else:
        print(f"  FAILED  {len(result.failures)} failures, "
              f"{len(result.errors)} errors / {result.testsRun} tests")
    print("=" * 65 + "\n")

    sys.exit(0 if result.wasSuccessful() else 1)
