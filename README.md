# Aluminum Alloy Mechanical Property Prediction  
Predicting YTS / UTS / EL for aluminum alloys using machine learning.

---

## 🚀 Project Status

This project **currently uses only a Random Forest model**.  
⚠️ This version is **for testing, debugging, and pipeline validation only**.  
More machine learning models (XGBoost, CatBoost, LightGBM, MLP, SVR, etc.) will be added later.

---

## 📘 Overview

The script `main.py` builds a machine-learning workflow to:

1. **Load and preprocess the aluminum alloy dataset**
2. **Automatically extract and filter features**
3. **Handle numerical and categorical variables**
4. **Train Random Forest models for each target (YTS / UTS / EL)**
5. **Predict values for samples labeled as `Unspecified`**
6. **Save predictions and evaluation metrics**

Outputs include:

- `unspecified_predictions.xlsx`
- `model_metrics.xlsx`

---

## 📂 Project Files
main.py # ML training and prediction script
dataset.xlsx # Input dataset (1–7 series alloys + Unspecified)
model_metrics.xlsx # Model performance (MAE, RMSE, R²)
unspecified_predictions.xlsx# Predicted mechanical properties

---

# 🧠 How the Code Works

Below is a breakdown of what each part of `main.py` does.

---

## 1. Import Libraries & Set Environment

Handles warnings and prepares ML tools.

```python
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
```
## 2. RMSE Compatibility (Different sklearn versions)
```python
try:
    from sklearn.metrics import root_mean_squared_error
    def RMSE(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:
    from sklearn.metrics import mean_squared_error
    def RMSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
```

Ensures compatibility across sklearn 1.0–1.4.

## 3. OneHotEncoder Compatibility (sparse vs sparse_output)
```python
import inspect
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)
```

Ensures OHE works regardless of sklearn version.

## 4. Load Dataset
```python
df = pd.read_excel("dataset.xlsx")
```
## 5. Extract Alloy Series (1–7)
```python
def get_series(x):
    try:
        return int(str(int(x))[0])
    except:
        return 0

df["Series"] = df["Alloy"].apply(get_series)
```
## 6. Split Training vs Unspecified Samples
```python
train_df = df[df["Alloy"] != "Unspecified"]
test_df  = df[df["Alloy"] == "Unspecified"]
```

Training samples = labeled alloys
Prediction samples = Unspecified alloys

## 7. Feature Selection
```python
drop_cols = ["YTS", "UTS", "EL", "Reference"]
feature_drop = ["Series", "Alloy"]
feature_cols = [c for c in df.columns if c not in drop_cols + feature_drop]
```

Removes:

Targets

Reference column

Series (redundant)

Alloy (avoids data leakage)

## 8. Preprocessing Pipelines
```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OHE),
])

preprocess = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])
```
## 9. Model Training (Random Forest)
```python
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
```

Each target (YTS / UTS / EL) is trained independently.

## 10. Prediction Loop
```python
for target in target_cols:
    ...
```

Stores predictions into unspecified_predictions.xlsx.

## 11. Save Outputs
unspecified_predictions.xlsx     # Predictions for Unspecified alloys
model_metrics.xlsx               # MAE / RMSE / R²

## 🔮 Future Updates

Planned model additions:

XGBoost

LightGBM

CatBoost

SVR

MLP neural networks

SHAP feature interpretation

Hyperparameter tuning

Cross-validation

Ensemble models

## 👤 Author

haitaobu18
GitHub: https://github.com/haitaobu18
