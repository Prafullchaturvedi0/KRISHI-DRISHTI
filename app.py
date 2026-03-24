"""
KrishiDrishti — Flask Web Server
==================================
Connects the ML inference engine (classifier.py) to the HTML web interface.
Serves the UI and handles image upload → model prediction → JSON response.

Routes:
  GET  /          → serve the main HTML interface
  POST /predict   → receive uploaded image, run inference, return JSON
  GET  /history   → return training history (for the loss curve charts in UI)
  GET  /health    → health check
"""

import os
import io
import json
import base64
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from PIL import Image

# ── Try to import model; gracefully handle missing model file ──────────────
MODEL_PATH = Path("model/best_model.pt")
HISTORY_PATH = Path("model/training_history.json")

try:
    from model.classifier import InferenceEngine, DISEASE_CLASSES, NUM_CLASSES
    if MODEL_PATH.exists():
        engine = InferenceEngine(str(MODEL_PATH))
        MODEL_LOADED = True
        print("✓ ML model loaded successfully")
    else:
        engine = None
        MODEL_LOADED = False
        print("⚠ No trained model found. Run train() first.")
        print("  Running in DEMO mode — will return simulated predictions.")
except ImportError as e:
    engine = None
    MODEL_LOADED = False
    print(f"⚠ PyTorch not installed: {e}. Running in DEMO mode.")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


# ─────────────────────────────────────────────────────────────────────────────
# DEMO MODE  (returns realistic simulated output when model is not trained yet)
# ─────────────────────────────────────────────────────────────────────────────
import random

DEMO_RESULTS = [
    {
        "top_class": "Wheat___Brown_rust",
        "confidence": 0.8734,
        "uncertain": False,
        "entropy": 0.6821,
        "is_healthy": False,
        "hindi_name": "गेहूं - भूरी फफूंद",
        "treatment": "Propiconazole 25% EC का 0.1% घोल छिड़कें। 15 दिन बाद दोहराएं।",
        "top3": [
            {"class": "Wheat___Brown_rust",    "probability": 0.8734, "hindi": "गेहूं - भूरी फफूंद"},
            {"class": "Wheat___Yellow_rust",   "probability": 0.0921, "hindi": "गेहूं - पीली फफूंद"},
            {"class": "Wheat___Powdery_mildew","probability": 0.0231, "hindi": "गेहूं - चूर्णिल आसिता"},
        ],
        "probabilities": {
            "Wheat___Brown_rust": 0.8734, "Wheat___Yellow_rust": 0.0921,
            "Wheat___Powdery_mildew": 0.0231, "Wheat___Healthy": 0.0114,
        }
    },
    {
        "top_class": "Cotton___Leaf_curl_virus",
        "confidence": 0.7612,
        "uncertain": False,
        "entropy": 1.2340,
        "is_healthy": False,
        "hindi_name": "कपास - पत्ती मुड़न वायरस",
        "treatment": "संक्रमित पौधे उखाड़ें। सफेद मक्खी नियंत्रण हेतु Imidacloprid 17.8 SL @ 0.5ml/L।",
        "top3": [
            {"class": "Cotton___Leaf_curl_virus",   "probability": 0.7612, "hindi": "कपास - पत्ती मुड़न वायरस"},
            {"class": "Cotton___Bacterial_blight",  "probability": 0.1543, "hindi": "कपास - जीवाणु झुलसा"},
            {"class": "Cotton___Alternaria_leaf_spot","probability": 0.0612, "hindi": "कपास - आल्टरनेरिया धब्बा"},
        ],
        "probabilities": {
            "Cotton___Leaf_curl_virus": 0.7612, "Cotton___Bacterial_blight": 0.1543,
            "Cotton___Alternaria_leaf_spot": 0.0612, "Cotton___healthy": 0.0233,
        }
    },
    {
        "top_class": "Tomato___healthy",
        "confidence": 0.9421,
        "uncertain": False,
        "entropy": 0.3102,
        "is_healthy": True,
        "hindi_name": "टमाटर - स्वस्थ",
        "treatment": "पौधा स्वस्थ दिखता है। नियमित देखभाल जारी रखें।",
        "top3": [
            {"class": "Tomato___healthy",      "probability": 0.9421, "hindi": "टमाटर - स्वस्थ"},
            {"class": "Tomato___Early_blight", "probability": 0.0321, "hindi": "टमाटर - अगेती झुलसा"},
            {"class": "Tomato___Leaf_Mold",    "probability": 0.0158, "hindi": "टमाटर - पत्ती मोल्ड"},
        ],
        "probabilities": {
            "Tomato___healthy": 0.9421, "Tomato___Early_blight": 0.0321,
            "Tomato___Leaf_Mold": 0.0158, "Tomato___Late_blight": 0.0100,
        }
    },
    {
        "top_class": "Soybean___Frogeye_leaf_spot",
        "confidence": 0.5234,
        "uncertain": True,
        "entropy": 2.8910,
        "is_healthy": False,
        "hindi_name": "सोयाबीन - मेंढक नेत्र धब्बा",
        "treatment": "Carbendazim 50% WP @ 1g/L या Thiophanate methyl का उपयोग करें।",
        "top3": [
            {"class": "Soybean___Frogeye_leaf_spot",    "probability": 0.5234, "hindi": "सोयाबीन - मेंढक नेत्र धब्बा"},
            {"class": "Soybean___Bacterial_pustule",    "probability": 0.2891, "hindi": "सोयाबीन - जीवाणु पस्च्यूल"},
            {"class": "Soybean___Sudden_death_syndrome","probability": 0.1343, "hindi": "सोयाबीन - अचानक मृत्यु"},
        ],
        "probabilities": {
            "Soybean___Frogeye_leaf_spot": 0.5234, "Soybean___Bacterial_pustule": 0.2891,
            "Soybean___Sudden_death_syndrome": 0.1343, "Soybean___healthy": 0.0532,
        }
    }
]

DEMO_HISTORY = {
    "history": {
        "train_loss": [1.82, 1.54, 1.31, 1.12, 0.98, 0.87, 0.79, 0.72, 0.66, 0.61,
                       0.57, 0.53, 0.50, 0.47, 0.45, 0.43, 0.41, 0.40, 0.39, 0.38],
        "val_loss":   [1.91, 1.62, 1.38, 1.19, 1.04, 0.94, 0.85, 0.79, 0.73, 0.68,
                       0.64, 0.61, 0.58, 0.56, 0.54, 0.53, 0.52, 0.52, 0.51, 0.51],
        "train_acc":  [0.41, 0.52, 0.60, 0.67, 0.72, 0.76, 0.79, 0.81, 0.83, 0.85,
                       0.86, 0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.91, 0.92, 0.92],
        "val_acc":    [0.39, 0.49, 0.58, 0.65, 0.70, 0.74, 0.77, 0.80, 0.82, 0.83,
                       0.85, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.89, 0.89, 0.90],
        "lr":         [0.01]*10 + [0.005]*5 + [0.0025]*5
    },
    "evaluation": {
        "accuracy": 0.9012, "macro_f1": 0.8843,
        "num_samples": 5400
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main HTML interface."""
    html_path = Path("templates/index.html")
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>KrishiDrishti</h1><p>Template not found.</p>", 404


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts: multipart/form-data with 'image' field (JPEG/PNG)
    Returns: JSON prediction result
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read and validate image
        image_bytes = file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Validate size
        w, h = pil_image.size
        if w < 32 or h < 32:
            return jsonify({"error": "Image too small (min 32×32)"}), 400

        # Encode thumbnail for display in response
        thumb = pil_image.copy()
        thumb.thumbnail((200, 200))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=75)
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        # Run inference
        if MODEL_LOADED and engine is not None:
            result = engine.predict(pil_image)
        else:
            # Demo mode: pick a random realistic result
            result = random.choice(DEMO_RESULTS).copy()
            result["demo_mode"] = True

        result["image_size"] = [w, h]
        result["thumbnail"]  = f"data:image/jpeg;base64,{thumb_b64}"

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def training_history():
    """Return training loss/accuracy history for chart rendering in the UI."""
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, "r") as f:
            return jsonify(json.load(f))
    return jsonify(DEMO_HISTORY)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED,
        "demo_mode": not MODEL_LOADED,
        "num_classes": NUM_CLASSES if MODEL_LOADED else 38
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  KrishiDrishti — Crop Disease Detection Server")
    print("="*55)
    print(f"  Model: {'Loaded ✓' if MODEL_LOADED else 'Demo mode (no trained model)'}")
    print(f"  Open: http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
