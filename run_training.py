import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.models.save_models import save_artifacts

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.pipelines.encoding_pipeline import EncodingPipeline

# 1. Load preprocessed data

X = pd.read_csv("data/02-preprocessed/X_preprocessed.csv")
y = pd.read_csv("data/02-preprocessed/y_preprocessed.csv").squeeze()

print(f"Dataset Shape: {X.shape}")


# 2. Train-test split (STRATIFIED)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Encoding (FIT ONLY ON TRAIN)

encoder = EncodingPipeline(cardinality_threshold=10)
encoder.fit(X_train, y_train)

X_train_enc = encoder.transform(X_train)
X_test_enc = encoder.transform(X_test)


# Helper functions

THRESHOLD = 0.35  # tuned for imbalance

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    print(f"\n================ {name} =================")

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_prob)
    test_auc = roc_auc_score(y_test, test_prob)

    test_pred = (test_prob >= THRESHOLD).astype(int)

    print(f"Train ROC-AUC : {train_auc:.4f}")
    print(f"Test  ROC-AUC : {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return test_prob, train_auc, test_auc


def cross_validate_model(name, model, X, y):
    print(f"\n--- Cross Validation: {name} ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC scores: {scores}")
    print(f"Mean CV ROC-AUC : {scores.mean():.4f}")
    return scores.mean()

# 4. Logistic Regression

log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="liblinear",
    C=0.5
)

log_reg.fit(X_train_enc, y_train)

lr_prob, lr_train_auc, lr_test_auc = evaluate_model(
    "Logistic Regression",
    log_reg,
    X_train_enc, y_train,
    X_test_enc, y_test
)

lr_cv_auc = cross_validate_model(
    "Logistic Regression",
    log_reg,
    X_train_enc,
    y_train
)


# 5. Random Forest

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_enc, y_train)

rf_prob, rf_train_auc, rf_test_auc = evaluate_model(
    "Random Forest",
    rf,
    X_train_enc, y_train,
    X_test_enc, y_test
)

rf_cv_auc = cross_validate_model(
    "Random Forest",
    rf,
    X_train_enc,
    y_train
)

# 6. XGBoost (VERSION SAFE)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train_enc, y_train)

xgb_prob, xgb_train_auc, xgb_test_auc = evaluate_model(
    "XGBoost",
    xgb,
    X_train_enc, y_train,
    X_test_enc, y_test
)

xgb_cv_auc = cross_validate_model(
    "XGBoost",
    xgb,
    X_train_enc,
    y_train
)

# 7. ROC Curve Comparison

plt.figure(figsize=(8, 6))

def plot_roc(y_true, y_prob, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=label)

plot_roc(y_test, lr_prob, "Logistic Regression")
plot_roc(y_test, rf_prob, "Random Forest")
plot_roc(y_test, xgb_prob, "XGBoost")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 8. Final Overfitting Summary

print("\n OVERFITTING CHECK ")
print(f"LogReg   | Train: {lr_train_auc:.3f} | Test: {lr_test_auc:.3f} | CV: {lr_cv_auc:.3f}")
print(f"RF       | Train: {rf_train_auc:.3f} | Test: {rf_test_auc:.3f} | CV: {rf_cv_auc:.3f}")
print(f"XGBoost  | Train: {xgb_train_auc:.3f} | Test: {xgb_test_auc:.3f} | CV: {xgb_cv_auc:.3f}")

# 9. Save final in src\models folder

save_artifacts(
    encoder=encoder,
    log_reg=log_reg,
    xgb_model=xgb
)

# 10. Save XGBoost test predictions (BUSINESS OUTPUT)

THRESHOLD = 0.35  # same threshold used in evaluation

xgb_test_predictions = pd.DataFrame({
    "actual_label": y_test.values,
    "predicted_probability": xgb_prob,
    "predicted_label": (xgb_prob >= THRESHOLD).astype(int)
})

output_path = "data/04-predictions/xgb_test_predictions.csv"
xgb_test_predictions.to_csv(output_path, index=False)

print(f" XGBoost test predictions saved to: {output_path}")