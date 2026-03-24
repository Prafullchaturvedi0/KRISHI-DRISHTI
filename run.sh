#!/usr/bin/env bash
# =============================================================================
#  KrishiDrishti — run.sh
#  One-command setup and launch script
#
#  Usage:
#    chmod +x run.sh          # make executable (first time only)
#    ./run.sh                 # install deps + start server in DEMO mode
#    ./run.sh --train         # install deps + run model training + start server
#    ./run.sh --test          # install deps + run test suite only
#    ./run.sh --port 8080     # start on a custom port (default: 5000)
#    ./run.sh --help          # show this help
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
banner()  { echo -e "\n${BOLD}$*${NC}\n"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
MODE="serve"          # serve | train | test
PORT=5000
VENV_DIR=".venv"
DATA_DIR="data/plantvillage"
MIN_PYTHON="3.9"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train)  MODE="train";  shift ;;
    --test)   MODE="test";   shift ;;
    --port)   PORT="$2";     shift 2 ;;
    --help|-h)
      echo ""
      echo "  KrishiDrishti — Crop Disease Detection"
      echo ""
      echo "  Usage: ./run.sh [OPTIONS]"
      echo ""
      echo "  Options:"
      echo "    (no flag)      Install dependencies and start the web server (DEMO mode)"
      echo "    --train        Train the ML model, then start the server"
      echo "    --test         Run the test suite and exit"
      echo "    --port PORT    Bind the server to PORT (default: 5000)"
      echo "    --help         Show this message"
      echo ""
      exit 0 ;;
    *) warn "Unknown argument: $1 (ignored)"; shift ;;
  esac
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}  ╔══════════════════════════════════════╗"
echo -e "  ║   KrishiDrishti — कृषि दृष्टि       ║"
echo -e "  ║   AI Crop Disease Detection          ║"
echo -e "  ╚══════════════════════════════════════╝${NC}"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
banner "Step 1 — Checking Python version"

if ! command -v python3 &>/dev/null; then
  error "python3 not found. Install Python ${MIN_PYTHON}+ from https://python.org"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
MIN_MAJOR=$(echo "$MIN_PYTHON" | cut -d. -f1)
MIN_MINOR=$(echo "$MIN_PYTHON" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt "$MIN_MAJOR" ]] || \
   { [[ "$PY_MAJOR" -eq "$MIN_MAJOR" ]] && [[ "$PY_MINOR" -lt "$MIN_MINOR" ]]; }; then
  error "Python ${PY_VERSION} found, but ${MIN_PYTHON}+ required."
fi
success "Python ${PY_VERSION} ✓"

# ── 2. Virtual environment ────────────────────────────────────────────────────
banner "Step 2 — Setting up virtual environment"

if [[ ! -d "$VENV_DIR" ]]; then
  info "Creating virtual environment at ${VENV_DIR}/ ..."
  python3 -m venv "$VENV_DIR"
  success "Virtual environment created."
else
  info "Virtual environment already exists at ${VENV_DIR}/ — reusing."
fi

# Activate
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
success "Virtual environment activated."

# ── 3. Install dependencies ───────────────────────────────────────────────────
banner "Step 3 — Installing dependencies"

pip install --upgrade pip --quiet
info "Installing from requirements.txt ..."
pip install -r requirements.txt --quiet
success "All dependencies installed."

# Check GPU availability (optional, informational)
GPU_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [[ "$GPU_AVAILABLE" == "True" ]]; then
  GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
  success "GPU detected: ${GPU_NAME} — training will be fast ⚡"
else
  warn "No GPU detected — running on CPU. Training will be slow; DEMO mode is instant."
fi

# ── 4. Validate project structure ─────────────────────────────────────────────
banner "Step 4 — Validating project structure"

REQUIRED_FILES=("app.py" "model/classifier.py" "templates/index.html" "requirements.txt")
for f in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    error "Required file missing: $f"
  fi
  success "${f} ✓"
done

# Create directories that must exist at runtime
mkdir -p model static data logs
success "Runtime directories ready."

# ── 5. Mode dispatch ──────────────────────────────────────────────────────────

# ── 5a. TEST mode ─────────────────────────────────────────────────────────────
if [[ "$MODE" == "test" ]]; then
  banner "Step 5 — Running test suite"
  info "Running test_basic.py ..."
  python3 -m pytest test_basic.py -v --tb=short 2>&1 | tee logs/test_$(date +%Y%m%d_%H%M%S).log
  TEST_EXIT=${PIPESTATUS[0]}
  if [[ "$TEST_EXIT" -eq 0 ]]; then
    success "All tests passed ✓"
  else
    error "Some tests failed. See logs/ for details."
  fi
  exit "$TEST_EXIT"
fi

# ── 5b. TRAIN mode ───────────────────────────────────────────────────────────
if [[ "$MODE" == "train" ]]; then
  banner "Step 5 — Training the ML model"

  if [[ ! -d "$DATA_DIR" ]]; then
    warn "Dataset not found at: ${DATA_DIR}/"
    echo ""
    echo "  To train the real model, download the PlantVillage dataset:"
    echo "  https://www.kaggle.com/datasets/emmarex/plantdisease"
    echo "  Extract it so the path looks like:"
    echo "    data/plantvillage/Apple___Apple_scab/image.jpg"
    echo ""
    echo "  Falling back to DEMO mode for now."
    echo ""
    MODE="serve"
  else
    info "Starting training (this may take 30–120 minutes on CPU) ..."
    python3 -c "
from model.classifier import train
train(
    data_dir='${DATA_DIR}',
    epochs=30,
    batch_size=32,
    dropout_rate=0.4,
    fine_tune_blocks=3,
    save_path='model/best_model.pt'
)
" 2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log
    success "Training complete. Model saved to model/best_model.pt"
  fi
fi

# ── 5c. SERVE mode (always runs after train, or standalone) ──────────────────
banner "Step 6 — Starting KrishiDrishti Web Server"

MODEL_STATUS="DEMO mode (no trained model)"
if [[ -f "model/best_model.pt" ]]; then
  MODEL_STATUS="Trained model loaded ✓"
fi

echo -e "  ${BOLD}URL:${NC}    http://localhost:${PORT}"
echo -e "  ${BOLD}Model:${NC}  ${MODEL_STATUS}"
echo -e "  ${BOLD}Stop:${NC}   Ctrl+C"
echo ""

# Log server startup
mkdir -p logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server starting on port ${PORT}" >> logs/server.log

python3 app.py 2>&1 | tee -a logs/server.log
