import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

# 1. Load data
df = pd.read_csv("data/churn.csv")

# 2. Target: Yes -> 1 (customer left), No -> 0 (customer stayed)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 3. Select features
features = ["tenure", "MonthlyCharges", "SeniorCitizen", "Partner", "Dependents",
            "Contract", "InternetService", "OnlineSecurity", "TechSupport",
            "PaperlessBilling", "PaymentMethod"]
X = df[features]
y = df["Churn"]

# 4. Split (stratify keeps the churn ratio equal in both sets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Preprocessing
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

# 6. Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# 7. Train and evaluate
results, trained = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    })
    trained[name] = pipe

results_df = pd.DataFrame(results).round(3)
print(results_df.to_string(index=False))

# 8. Save the best model (by F1 on the churn class)
best_name = results_df.sort_values("F1", ascending=False).iloc[0]["Model"]
joblib.dump(trained[best_name], "models/model.pkl")
print("\nBest model:", best_name)

# 9. Save confusion matrix for the README
ConfusionMatrixDisplay.from_estimator(trained[best_name], X_test, y_test,
                                      display_labels=["Stayed", "Churned"])
plt.title(f"Confusion Matrix - {best_name}")
plt.savefig("images/confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved model and confusion matrix.")