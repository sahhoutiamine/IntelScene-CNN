
#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-.}"

mkdir -p \
    "$ROOT_DIR/app" \
    "$ROOT_DIR/data/intel_image_classification" \
    "$ROOT_DIR/data/splits" \
    "$ROOT_DIR/notebooks" \
    "$ROOT_DIR/models" \
    "$ROOT_DIR/reports/figures" \
    "$ROOT_DIR/reports/metrics" \
    "$ROOT_DIR/src/data" \
    "$ROOT_DIR/src/models" \
    "$ROOT_DIR/src/training" \
    "$ROOT_DIR/src/evaluation" \
    "$ROOT_DIR/src/prediction" \
    "$ROOT_DIR/src/utils" \
    "$ROOT_DIR/tests"

touch \
    "$ROOT_DIR/README.md" \
    "$ROOT_DIR/requirements.txt" \
    "$ROOT_DIR/app/app.py" \
    "$ROOT_DIR/src/__init__.py" \
    "$ROOT_DIR/src/data/__init__.py" \
    "$ROOT_DIR/src/data/explore_dataset.py" \
    "$ROOT_DIR/src/data/prepare_dataset.py" \
    "$ROOT_DIR/src/models/__init__.py" \
    "$ROOT_DIR/src/models/cnn.py" \
    "$ROOT_DIR/src/training/__init__.py" \
    "$ROOT_DIR/src/training/train.py" \
    "$ROOT_DIR/src/evaluation/__init__.py" \
    "$ROOT_DIR/src/evaluation/evaluate.py" \
    "$ROOT_DIR/src/prediction/__init__.py" \
    "$ROOT_DIR/src/prediction/predict.py" \
    "$ROOT_DIR/src/utils/__init__.py" \
    "$ROOT_DIR/src/utils/constants.py" \
    "$ROOT_DIR/tests/test_data.py" \
    "$ROOT_DIR/tests/test_prediction.py"

touch \
    "$ROOT_DIR/data/intel_image_classification/.gitkeep" \
    "$ROOT_DIR/data/splits/.gitkeep" \
    "$ROOT_DIR/models/.gitkeep" \
    "$ROOT_DIR/reports/figures/.gitkeep" \
    "$ROOT_DIR/reports/metrics/.gitkeep"

if [[ ! -f "$ROOT_DIR/.gitignore" ]]; then
    cat > "$ROOT_DIR/.gitignore" <<'EOF'
__pycache__/
*.py[cod]
.venv/
venv/
.env
.ipynb_checkpoints/

data/intel_image_classification/*
!data/intel_image_classification/.gitkeep

data/splits/*
!data/splits/.gitkeep

data/image_audit.csv
models/*
!models/.gitkeep
EOF
fi

if [[ ! -s "$ROOT_DIR/requirements.txt" ]]; then
    cat > "$ROOT_DIR/requirements.txt" <<'EOF'
tensorflow
pandas
matplotlib
Pillow
scikit-learn
streamlit
EOF
fi

echo "Project structure created successfully in: $ROOT_DIR"