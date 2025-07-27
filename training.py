import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Step 1: Load dataset
df = pd.read_csv("creditcard.csv")
print("✅ Dataset loaded")

# Step 2: Preprocess - Drop 'Time', Scale 'Amount'
df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
df = df.drop(["Time"], axis=1)
print("✅ Preprocessing complete")

# Step 3: Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Step 4: Handle class imbalance (Under-sampling)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)
print("✅ Applied Random UnderSampling")

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Step 6: Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)
print("✅ Model training complete")

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

# a) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as confusion_matrix.png")

# b) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_score = roc_auc_score(y_test, y_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_score:.2f})", color="darkorange")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
print("✅ ROC curve saved as roc_curve.png")

# c) Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_score)

# Step 8: Save the model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as xgb_model.pkl")
