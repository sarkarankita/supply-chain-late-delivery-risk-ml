import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import argparse
import pandas as pd
import joblib

# IMPORTANT: this import is REQUIRED for pickle loading

from src.pipelines.encoding_pipeline import EncodingPipeline

# =========================================================
# Paths
# =========================================================
MODEL_DIR = Path("src/models")
OUTPUT_DIR = Path("data/04-predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_PATH = MODEL_DIR / "encoder.pkl"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"

THRESHOLD = 0.35


# =========================================================
# Load artifacts
# =========================================================
def load_artifacts():
    encoder = joblib.load(ENCODER_PATH)
    model = joblib.load(MODEL_PATH)
    return encoder, model


# =========================================================
# Predict function
# =========================================================
def predict(input_csv: str, output_csv: str):
    print("🔹 Loading input data...")
    data = pd.read_csv(input_csv)

    print("🔹 Loading model & encoder...")
    encoder, model = load_artifacts()

    print("🔹 Encoding data...")
    X_encoded = encoder.transform(data)

    print("🔹 Generating predictions...")
    prob = model.predict_proba(X_encoded)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)

    results = data.copy()
    results["predicted_probability"] = prob
    results["predicted_label"] = pred

    results.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to: {output_csv}")


# =========================================================
# CLI Entry
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Late Delivery Risk Prediction")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR / "predictions.csv"),
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    predict(args.input, args.output)
