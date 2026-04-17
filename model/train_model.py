import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


RANDOM_STATE = 42


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    height_m = df["height"] / 100.0
    df["bmi"] = df["weight"] / np.where(height_m > 0, height_m**2, np.nan)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["map"] = df["ap_lo"] + (df["pulse_pressure"] / 3.0)
    df["age_bp_interaction"] = df["age"] * df["ap_hi"]
    df["lifestyle_risk"] = df[["smoke", "alco"]].sum(axis=1) + (1 - df["active"])
    return df


def clean_medical_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Dataset stores age in days; convert to years for model usability.
    df["age"] = df["age"] / 365.0

    # Convert clearly invalid values to NaN so imputation can handle them.
    bounds = {
        "age": (18, 100),
        "height": (120, 230),
        "weight": (30, 250),
        "ap_hi": (70, 260),
        "ap_lo": (40, 160),
    }

    for col, (low, high) in bounds.items():
        df[col] = df[col].where(df[col].between(low, high), np.nan)

    invalid_bp = df["ap_hi"] <= df["ap_lo"]
    df.loc[invalid_bp, ["ap_hi", "ap_lo"]] = np.nan

    for col in ["cholesterol", "gluc"]:
        df[col] = df[col].where(df[col].isin([1, 2, 3]), np.nan)

    for col in ["gender", "smoke", "alco", "active", "cardio"]:
        df[col] = df[col].where(df[col].isin([0, 1]), np.nan)

    return df


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def performance_recommendation(accuracy: float) -> str:
    if accuracy < 0.70:
        return (
            "Major improvement needed: audit data quality, add stronger feature engineering, "
            "address class imbalance, and evaluate tree-boosting models."
        )
    if accuracy < 0.85:
        return (
            "Moderate improvement opportunity: tune hyperparameters, tighten validation, "
            "and optimize prediction threshold for recall."
        )
    return (
        "Strong baseline: focus on calibration, regularization fine-tuning, and threshold "
        "selection based on clinical false-negative tolerance."
    )


def find_threshold_for_recall(y_true: pd.Series, y_proba: np.ndarray, target_recall: float = 0.85) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.5
    best_idx = valid_idx[np.argmax(precision[:-1][valid_idx])]
    return float(thresholds[best_idx])


def save_visualizations(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()


def save_feature_importance(model_pipeline: Pipeline, feature_names: list[str], output_dir: Path) -> None:
    classifier = model_pipeline.named_steps["model"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df.head(12), x="importance", y="feature", palette="viridis")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()


def dataset_quality_checks(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    invalid_bp_ratio = float((df["ap_hi"] <= df["ap_lo"]).mean()) if total_rows else 0.0
    missing_ratio = float(df.isna().mean().mean()) if total_rows else 0.0
    class_ratio = float(df["cardio"].mean()) if "cardio" in df.columns and total_rows else 0.0
    return {
        "rows": total_rows,
        "invalid_bp_ratio": invalid_bp_ratio,
        "missing_ratio": missing_ratio,
        "cardio_positive_ratio": class_ratio,
    }


def save_logistic_diagnostics(
    trained_models: dict[str, Pipeline],
    feature_columns: list[str],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {}

    logistic_pipeline = trained_models.get("Logistic Regression")
    if logistic_pipeline is None:
        return diagnostics

    logistic_model = logistic_pipeline.named_steps["model"]
    if not hasattr(logistic_model, "coef_"):
        return diagnostics

    coef = logistic_model.coef_[0]
    coef_df = pd.DataFrame({"feature": feature_columns, "coefficient": coef})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(output_dir / "logistic_coefficients.csv", index=False)

    diagnostics["max_abs_coefficient"] = float(np.max(np.abs(coef)))
    diagnostics["top_coefficients"] = coef_df.head(8)[["feature", "coefficient"]].to_dict(orient="records")
    return diagnostics


def sanity_profile_predictions(
    trained_models: dict[str, Pipeline],
    feature_columns: list[str],
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_profiles = pd.DataFrame(
        [
            {
                "profile": "healthy_reference",
                "age": 32,
                "gender": 0,
                "height": 168,
                "weight": 62,
                "ap_hi": 118,
                "ap_lo": 76,
                "cholesterol": 1,
                "gluc": 1,
                "smoke": 0,
                "alco": 0,
                "active": 1,
            },
            {
                "profile": "high_risk_reference",
                "age": 67,
                "gender": 1,
                "height": 165,
                "weight": 92,
                "ap_hi": 168,
                "ap_lo": 102,
                "cholesterol": 3,
                "gluc": 3,
                "smoke": 1,
                "alco": 1,
                "active": 0,
            },
        ]
    )

    profile_labels = base_profiles["profile"].tolist()
    profile_features = add_engineered_features(base_profiles.drop(columns=["profile"]))
    profile_features = profile_features[feature_columns]

    rows = []
    for model_name, pipeline in trained_models.items():
        probs = pipeline.predict_proba(profile_features)[:, 1]
        for label, prob in zip(profile_labels, probs):
            rows.append({"model": model_name, "profile": label, "probability": float(prob)})

    pd.DataFrame(rows).to_csv(output_dir / "sanity_profile_predictions.csv", index=False)
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "cardio_train.csv"
    plots_dir = project_root / "model" / "plots"

    raw_df = pd.read_csv(data_path, sep=";")
    quality_report = dataset_quality_checks(raw_df)

    df = clean_medical_data(raw_df)
    df = add_engineered_features(df)
    df = df.dropna(subset=["cardio"]).copy()

    feature_columns = [
        "age",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "bmi",
        "pulse_pressure",
        "map",
        "age_bp_interaction",
        "lifestyle_risk",
    ]

    X = df[feature_columns]
    y = df["cardio"].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    preprocessor = build_preprocessor(feature_columns)

    model_candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=0.5,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_split=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }
    results = []
    trained_models: dict[str, Pipeline] = {}

    print("Class distribution (cardio=1 ratio):", round(float(y.mean()), 4))
    print("Raw data quality checks:")
    for key, value in quality_report.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")

    for model_name, estimator in model_candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_val, y_val_pred),
            "precision": precision_score(y_val, y_val_pred, zero_division=0),
            "recall": recall_score(y_val, y_val_pred, zero_division=0),
            "f1": f1_score(y_val, y_val_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_val_proba),
        }

        results.append(metrics)
        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by=["f1", "roc_auc"], ascending=False)
    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    # Refit the best model on the full training split before final test evaluation.
    best_model.fit(X_train_full, y_train_full)

    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    threshold = find_threshold_for_recall(y_test, y_proba_best, target_recall=0.85)
    y_pred_threshold = (y_proba_best >= threshold).astype(int)

    final_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_threshold),
        "precision": precision_score(y_test, y_pred_threshold, zero_division=0),
        "recall": recall_score(y_test, y_pred_threshold, zero_division=0),
        "f1": f1_score(y_test, y_pred_threshold, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba_best),
    }

    print("\nModel Comparison (sorted by F1, ROC-AUC):")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model selected: {best_model_name}")
    print(f"Threshold tuned for recall>=0.85: {threshold:.3f}")
    print("\nFinal Metrics on test set:")
    for key, value in final_metrics.items():
        print(f"- {key}: {value:.4f}")
    print("\nRecommendation:")
    print(performance_recommendation(final_metrics["accuracy"]))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_threshold))

    save_visualizations(y_test, y_pred_threshold, y_proba_best, plots_dir)
    save_feature_importance(best_model, feature_columns, plots_dir)
    logistic_diag = save_logistic_diagnostics(trained_models, feature_columns, plots_dir)
    profile_debug = sanity_profile_predictions(trained_models, feature_columns, plots_dir)

    model_path = project_root / "heart_model.pkl"
    report_path = project_root / "model_report.json"

    joblib.dump(best_model, model_path)

    report = {
        "best_model": best_model_name,
        "threshold": threshold,
        "features": feature_columns,
        "metrics": final_metrics,
        "model_comparison": results_df.to_dict(orient="records"),
        "recommendation": performance_recommendation(final_metrics["accuracy"]),
        "data_quality": quality_report,
        "logistic_diagnostics": logistic_diag,
        "sanity_profile_predictions": profile_debug,
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved artifacts:")
    print(f"- Model: {model_path}")
    print(f"- Report: {report_path}")
    print(f"- Plots: {plots_dir}")


if __name__ == "__main__":
    main()
