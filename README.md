# Cardiovascular Disease Prediction Model

An end-to-end machine learning project for cardiovascular disease risk prediction using scikit-learn and a Streamlit user interface.

The system is designed as a screening tool, not a medical diagnosis system. It accepts common patient inputs such as age, blood pressure, cholesterol, glucose, smoking, alcohol intake, physical activity, height, and weight, then returns a risk probability and a Low / Medium / High risk category.

## Project Summary

- Problem type: binary classification
- Primary goal: predict cardiovascular disease risk from structured health features
- Current best model: Gradient Boosting
- Decision threshold: `0.3358`
- Training dataset size: `70,000` rows
- Positive class ratio: `0.4997`

## Features Used

Raw input features:

- Age
- Gender
- Height
- Weight
- Systolic blood pressure (`ap_hi`)
- Diastolic blood pressure (`ap_lo`)
- Cholesterol
- Glucose
- Smoking
- Alcohol intake
- Physical activity

Engineered features:

- BMI
- Pulse pressure
- Mean arterial pressure
- Age x systolic BP interaction
- Lifestyle risk score

## Model Comparison

The training pipeline compares multiple models and selects the best one based on validation performance.

Current evaluation summary from `model_report.json`:

- Gradient Boosting:
  - Accuracy: `0.7288`
  - Precision: `0.7453`
  - Recall: `0.6947`
  - F1-score: `0.7191`
  - ROC-AUC: `0.7978`
- Logistic Regression:
  - Accuracy: `0.7271`
  - Precision: `0.7466`
  - Recall: `0.6870`
  - F1-score: `0.7155`
  - ROC-AUC: `0.7901`
- Random Forest:
  - Accuracy: `0.7161`
  - Precision: `0.7258`
  - Recall: `0.6939`
  - F1-score: `0.7095`
  - ROC-AUC: `0.7785`

## Repository Structure

```text
Cardiovascular Disease Prediction Model/
├── app.py
├── data/
│   └── cardio_train.csv
├── heart_model.pkl
├── model_report.json
├── model/
│   ├── train_model.py
│   └── plots/
├── requirements.txt
├── scaler.pkl
└── README.md
```

## Setup

### 1) Create and activate a Python environment

On Windows PowerShell:

```powershell
cd "C:\Users\Mansi Arora\OneDrive\Desktop\ml projects\Cardiovascular Disease Prediction Model"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
py -m pip install -r requirements.txt
```

## Train the Model

Run the training script to clean data, engineer features, compare models, evaluate metrics, and save artifacts.

```powershell
py model\train_model.py
```

This generates or updates:

- `heart_model.pkl`
- `model_report.json`
- plots in `model/plots/`

## Run the Streamlit App

Launch the interactive prediction UI:

```powershell
py -m streamlit run app.py
```

## How the App Works

The Streamlit app:

- Collects patient inputs through a structured form
- Validates basic clinical constraints such as systolic BP being higher than diastolic BP
- Applies the same feature engineering and preprocessing logic used in training
- Produces a probability score
- Converts the score into a risk category:
  - Low risk
  - Medium risk
  - High risk

## Output Artifacts

The training pipeline also saves debugging and evaluation outputs:

- `model_report.json` - full training summary and metrics
- `model/plots/confusion_matrix.png`
- `model/plots/roc_curve.png`
- `model/plots/feature_importance.png`
- `model/plots/logistic_coefficients.csv`
- `model/plots/sanity_profile_predictions.csv`

## Debugging Notes

If the model predicts unusually high probabilities for normal inputs, check the following:

- whether the same preprocessing is used during training and inference
- whether the input values are realistic
- whether the dataset contains invalid blood pressure or other impossible medical values
- whether the threshold is too low for your chosen risk banding
- whether the trained model is being loaded from the latest artifact files

The current pipeline includes data quality checks and sanity profile predictions to help with this.

## Limitations

This project is useful for learning and screening, but it has important limitations:

- The dataset is manually created and may not reflect real hospital distributions
- It may contain bias or unrealistic correlations
- It has not been clinically validated
- It should not be used as a medical diagnostic system

## Future Improvements

- Train on a real public healthcare dataset
- Add probability calibration
- Add subgroup fairness checks
- Perform external validation on a separate cohort
- Explore clinical time-series data if available

## License

See `LICENSE` for license information.