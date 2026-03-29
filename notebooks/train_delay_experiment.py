# Databricks notebook source
import pandas as pd
import csv

# Load CSV
df = pd.read_csv("merged_output.csv")

# Show first 5 rows
print(df.head())
print(df.columns)   # column names
print(df.shape)     # rows, columns
print(df.info())    #

# COMMAND ----------

df = df[(df["delay_minutes"] >= -120) & (df["delay_minutes"] <= 600)]
print(df.head())

with open('top_100_trains_delay_dataset.csv', 'r') as f:
    print(df.columns)   # column names
    print(df.shape)     # rows, columns
    print(df.info())    # data types

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("top_100_trains_delay_dataset.csv")

df["Date"] = pd.to_datetime(df["Date"])

# time features
df["month_num"] = df["Date"].dt.month
df["month_name"] = df["Date"].dt.strftime("%b")
df["weekday_num"] = df["Date"].dt.dayofweek
df["weekday_name"] = df["Date"].dt.strftime("%a")

# =====================================
# PREPARE AGGREGATIONS
# =====================================

# 1) month-wise avg delay
monthly_delay = (
    df.groupby(["month_num", "month_name"])["delay_minutes"]
    .mean()
    .reset_index()
    .sort_values("month_num")
)

# 2) weekday-wise avg delay
weekday_delay = (
    df.groupby(["weekday_num", "weekday_name"])["delay_minutes"]
    .mean()
    .reset_index()
    .sort_values("weekday_num")
)

# 3) top delayed stations
station_delay = (
    df.groupby("station")["delay_minutes"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# 4) top delayed trains
train_delay = (
    df.groupby("train_no")["delay_minutes"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# 5) monthly trend of total delay count
monthly_volume = (
    df.groupby(["month_num", "month_name"])
    .size()
    .reset_index(name="records")
    .sort_values("month_num")
)

# =====================================
# SINGLE FIGURE DASHBOARD
# =====================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# -------------------------------------
# 1. month-wise avg delay
# -------------------------------------
axes[0, 0].plot(
    monthly_delay["month_name"],
    monthly_delay["delay_minutes"]
)
axes[0, 0].set_title("Average Delay by Month")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Delay (min)")

# -------------------------------------
# 2. weekday-wise
# -------------------------------------
axes[0, 1].plot(
    weekday_delay["weekday_name"],
    weekday_delay["delay_minutes"]
)
axes[0, 1].set_title("Average Delay by Weekday")
axes[0, 1].set_xlabel("Day")
axes[0, 1].set_ylabel("Delay (min)")

# -------------------------------------
# 3. top stations
# -------------------------------------
axes[1, 0].bar(
    station_delay.index,
    station_delay.values
)
axes[1, 0].set_title("Top 10 Delay-Prone Stations")
axes[1, 0].set_xlabel("Station")
axes[1, 0].set_ylabel("Avg Delay")
axes[1, 0].tick_params(axis="x", rotation=45)

# -------------------------------------
# 4. top trains
# -------------------------------------
axes[1, 1].bar(
    train_delay.index.astype(str),
    train_delay.values
)
axes[1, 1].set_title("Top 10 Delay-Prone Trains")
axes[1, 1].set_xlabel("Train")
axes[1, 1].set_ylabel("Avg Delay")
axes[1, 1].tick_params(axis="x", rotation=45)

# -------------------------------------
# 5. delay distribution
# -------------------------------------
axes[2, 0].hist(df["delay_minutes"], bins=50)
axes[2, 0].set_title("Delay Distribution")
axes[2, 0].set_xlabel("Delay")
axes[2, 0].set_ylabel("Frequency")

# -------------------------------------
# 6. monthly record volume
# -------------------------------------
axes[2, 1].plot(
    monthly_volume["month_name"],
    monthly_volume["records"]
)
axes[2, 1].set_title("Monthly Data Volume")
axes[2, 1].set_xlabel("Month")
axes[2, 1].set_ylabel("Records")

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Install libraries
# MAGIC %pip install optuna category_encoders lightgbm --quiet

# COMMAND ----------

# DBTITLE 1,Restart Python kernel
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Preprocessing & feature engineering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import lightgbm as lgb
import optuna
import mlflow
import mlflow.lightgbm
import warnings
warnings.filterwarnings("ignore")

# =============================================
# 1. LOAD & PREPROCESS DATA
# =============================================
df = pd.read_csv("merged_output.csv")

# Filter outliers (same as Cell 2)
df = df[(df["delay_minutes"] >= -120) & (df["delay_minutes"] <= 600)]
df = df.dropna(subset=["delay_minutes"])

# Parse date and engineer features
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
df["is_holiday"] = df["holiday"].notna().astype(int)



# Encode categorical features
le_station = LabelEncoder()
df["station_enc"] = le_station.fit_transform(df["station"])

le_day = LabelEncoder()
df["day_enc"] = le_day.fit_transform(df["day"])

# =============================================
# 2. FEATURE & TARGET SELECTION
# =============================================
FEATURES = [
    "train_no", "station_enc", "day_enc",
    "month", "day_of_week", "is_holiday"
]
TARGET = "delay_minutes"

X = df[FEATURES]
y = df[TARGET]

print(f"Dataset shape: {X.shape}")
print(f"Features: {FEATURES}")
print(f"Target stats:\n{y.describe()}")

# =============================================
# 3. CORRELATION ANALYSIS
# =============================================
corr_df = pd.concat([X, y], axis=1).corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Train / Validation / Test split
# =============================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# =============================================
# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Train : {X_train.shape[0]:,} rows")
print(f"Val   : {X_val.shape[0]:,} rows")
print(f"Test  : {X_test.shape[0]:,} rows")

# LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

# COMMAND ----------

# DBTITLE 1,Optuna hyperparameter tuning
# =============================================
# 5. HYPERPARAMETER TUNING WITH OPTUNA
# =============================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Recreate datasets so feature_pre_filter can be set
train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))

study = optuna.create_study(direction="minimize", study_name="train_delay_lgbm")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\nBest RMSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# COMMAND ----------

# DBTITLE 1,Train final model & log to MLflow
# =============================================
# 6. TRAIN FINAL MODEL & LOG WITH MLFLOW
# =============================================
best_params = study.best_params
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "boosting_type": "gbdt",
})

mlflow.set_experiment("/Users/lopamudra.wncc@gmail.com/train_delay/train_delay_experiment")

with mlflow.start_run(run_name="lgbm_best") as run:
    # Train final model
    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # Predictions
    y_train_pred = final_model.predict(X_train)
    y_val_pred   = final_model.predict(X_val)
    y_test_pred  = final_model.predict(X_test)

    # Metrics
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "train_mae":  mean_absolute_error(y_train, y_train_pred),
        "train_r2":   r2_score(y_train, y_train_pred),
        "val_rmse":   np.sqrt(mean_squared_error(y_val, y_val_pred)),
        "val_mae":    mean_absolute_error(y_val, y_val_pred),
        "val_r2":     r2_score(y_val, y_val_pred),
        "test_rmse":  np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "test_mae":   mean_absolute_error(y_test, y_test_pred),
        "test_r2":    r2_score(y_test, y_test_pred),
    }

    # Log params, metrics, model
    mlflow.log_params(best_params)
    mlflow.log_metrics(metrics)

    signature = mlflow.models.infer_signature(X_train, y_train_pred)
    mlflow.lightgbm.log_model(
        final_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
    )

    print("\n===== MODEL PERFORMANCE =====")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")
    print(f"\n  MLflow Run ID : {run.info.run_id}")

# COMMAND ----------

# DBTITLE 1,Evaluation plots & residual analysis
# =============================================
# 7. EVALUATION PLOTS
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7a. Predicted vs Actual
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.1, s=5)
axes[0, 0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", lw=2
)
axes[0, 0].set_xlabel("Actual Delay (min)")
axes[0, 0].set_ylabel("Predicted Delay (min)")
axes[0, 0].set_title("Predicted vs Actual")

# 7b. Residual Distribution
residuals = y_test - y_test_pred
axes[0, 1].hist(residuals, bins=80, edgecolor="black")
axes[0, 1].axvline(0, color="red", linestyle="--")
axes[0, 1].set_xlabel("Residual (min)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title(f"Residuals  (mean={residuals.mean():.2f}, std={residuals.std():.2f})")

# 7c. Feature Importance
importance = final_model.feature_importance(importance_type="gain")
feat_imp = pd.Series(importance, index=final_model.feature_name()).sort_values()
feat_imp.plot.barh(ax=axes[1, 0])
axes[1, 0].set_title("Feature Importance (Gain)")
axes[1, 0].set_xlabel("Importance")

# 7d. Residual Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Residual Q-Q Plot")

plt.tight_layout()
plt.show()

print(f"\nResidual mean: {residuals.mean():.4f}")
print(f"Residual std:  {residuals.std():.4f}")