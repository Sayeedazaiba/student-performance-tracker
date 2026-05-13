# ==========================================
# STUDENT PERFORMANCE TRACKER USING REGRESSION
# ==========================================

import warnings, os, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- CONFIG ----------------
DATA = "StudentPerformanceFactors.csv"
OUT  = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA)

print("\nSTUDENT PERFORMANCE TRACKER")
print("Rows:", df.shape[0], "| Columns:", df.shape[1])

# ---------------- PREPROCESSING ----------------
df.fillna(df.median(numeric_only=True), inplace=True)

# Fill categorical missing values
for c in df.select_dtypes(include="object"):
    df[c].fillna(df[c].mode()[0], inplace=True)

# Ordinal encoding
maps = {
    "Parental_Involvement": {"Low":0,"Medium":1,"High":2},
    "Access_to_Resources": {"Low":0,"Medium":1,"High":2},
    "Motivation_Level": {"Low":0,"Medium":1,"High":2},
    "Family_Income": {"Low":0,"Medium":1,"High":2},
    "Teacher_Quality": {"Low":0,"Medium":1,"High":2},
    "Parental_Education_Level": {"High School":0,"College":1,"Postgraduate":2},
    "Distance_from_Home": {"Near":0,"Moderate":1,"Far":2},
    "Peer_Influence": {"Negative":0,"Neutral":1,"Positive":2},
}

for col, mp in maps.items():
    if col in df:
        df[col] = df[col].map(mp)

# Label encoding
le = LabelEncoder()
for col in df.select_dtypes(include="object"):
    df[col] = le.fit_transform(df[col].astype(str))

# ---------------- SPLIT DATA ----------------
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---------------- MODELS ----------------
models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoost": GradientBoostingRegressor(random_state=42)
}

results = {}

print("\nMODEL PERFORMANCE\n")

for name, model in models.items():

    # Linear models use scaled data
    if name in ["Linear", "Ridge", "Lasso"]:
        Xtr, Xte = X_train_s, X_test_s
    else:
        Xtr, Xte = X_train, X_test

    model.fit(Xtr, y_train)
    pred = model.predict(Xte)

    r2   = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    cv   = cross_val_score(model, Xtr, y_train, cv=5, scoring="r2").mean()

    results[name] = [r2, rmse, mae, cv, pred, model]

    print(f"{name:15} R2={r2:.4f} RMSE={rmse:.2f} MAE={mae:.2f} CV={cv:.4f}")

# ---------------- BEST MODEL ----------------
best = max(results, key=lambda x: results[x][0])

r2, rmse, mae, cv, pred, best_model = results[best]

print(f"\nBest Model: {best}")
print(f"R2 Score : {r2:.4f}")

# Save model
joblib.dump(best_model, f"{OUT}/best_model.pkl")

# ---------------- VISUALIZATION ----------------
# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/heatmap.png")
plt.close()

# Actual vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title(f"{best} - Actual vs Predicted")
plt.tight_layout()
plt.savefig(f"{OUT}/prediction.png")
plt.close()

# Feature Importance / Coefficients
if hasattr(best_model, "feature_importances_"):
    imp = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    imp = pd.Series(best_model.coef_, index=X.columns)

imp.sort_values().plot(kind="barh", figsize=(8,6))
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/features.png")
plt.close()

# ---------------- SAMPLE PREDICTION ----------------
sample = X.iloc[[0]]

if best in ["Linear", "Ridge", "Lasso"]:
    score = best_model.predict(scaler.transform(sample))[0]
else:
    score = best_model.predict(sample)[0]

print(f"\nPredicted Exam Score: {score:.2f}")

print(f"\nAll outputs saved in '{OUT}' folder")