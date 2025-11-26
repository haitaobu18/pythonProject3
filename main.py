import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
    def RMSE(y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
except ImportError:
    from sklearn.metrics import mean_squared_error
    def RMSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)


import inspect
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)


data_path = "dataset.xlsx"
df = pd.read_excel(data_path)

# 提取 Series （铝合金 1–7 系）
def get_series(x):
    try:
        return int(str(int(x))[0])
    except:
        return 0

df["Series"] = df["Alloy"].apply(get_series)

# 划分训练/测试
train_df = df[df["Alloy"] != "Unspecified"].copy()
test_df  = df[df["Alloy"] == "Unspecified"].copy()

print("训练集行数:", train_df.shape[0])
print("测试集行数:", test_df.shape[0])

# 目标列
target_cols = ["YTS", "UTS", "EL"]
drop_cols = target_cols + ["Reference"]
#去掉干扰特征
feature_drop = ["Series", "Alloy"]

# 特征列
feature_cols = [c for c in df.columns if c not in drop_cols + feature_drop]


numeric_features = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in feature_cols if c not in numeric_features]

print("数值特征:", numeric_features)
print("类别特征:", categorical_features)


# =====================================================
# 预处理 Pipeline
# =====================================================
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


# =====================================================
# 单目标训练函数
# =====================================================
def train_predict_one_target(target, train_df, test_df, feature_cols):

    train_sub = train_df.dropna(subset=[target])  # 删除无标签行

    X_train = train_sub[feature_cols]
    y_train = train_sub[target]

    X_test = test_df[feature_cols]
    y_test = test_df[target]

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("scaler", StandardScaler()),
        ("model", rf)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # 只在测试集有真实值时评估指标
    mask = ~y_test.isna()
    if mask.sum() > 0:
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        rmse = RMSE(y_test[mask], y_pred[mask])
        r2 = r2_score(y_test[mask], y_pred[mask])
    else:
        mae = rmse = r2 = np.nan

    return pipe, y_pred, (mae, rmse, r2)


# =====================================================
# 针对 YTS / UTS / EL 循环训练
# =====================================================
results_df = test_df.copy()
metrics = {}

for target in target_cols:
    print(f"\n===== 训练目标: {target} =====")
    model, preds, (mae, rmse, r2) = train_predict_one_target(
        target, train_df, test_df, feature_cols
    )

    results_df[f"{target}_pred"] = preds
    metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"{target}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")


# =====================================================
# 保存结果
# =====================================================
results_df.to_excel("unspecified_predictions.xlsx", index=False)
pd.DataFrame(metrics).T.to_excel("model_metrics.xlsx")

print("\n所有预测已保存到 unspecified_predictions.xlsx")
print("模型指标已保存到 model_metrics.xlsx")
