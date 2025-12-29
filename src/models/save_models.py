import joblib
from pathlib import Path

# Save trained models & encoder

def save_artifacts(encoder, log_reg, xgb_model):
    model_dir = Path("src/models")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, model_dir / "encoder.pkl")
    joblib.dump(log_reg, model_dir / "logistic_model.pkl")
    joblib.dump(xgb_model, model_dir / "xgboost_model.pkl")

    print("Models and encoder saved successfully.")
