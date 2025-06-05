import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ----------------------------------------
# 📥 Load and Clean Data
# ----------------------------------------

df = pd.read_csv(r'C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\Telco-Customer-Churn.csv')
df.drop(['customerID'], axis=1, inplace=True)

# Handle empty values in TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Replace spaces with underscores
df = df.replace(' ', '_', regex=True)

# ----------------------------------------
# 🎯 Feature and Target Setup
# ----------------------------------------

X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})  # Binary labels

X = pd.get_dummies(X, columns=[
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
])

# ----------------------------------------
# 🧪 Train-Test Split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------
# 🚀 Train XGBoost Model
# ----------------------------------------

model = XGBClassifier(
    objective='binary:logistic',
    gamma=0.25,
    learning_rate=0.1,
    max_depth=4,
    reg_lambda=10,
    scale_pos_weight=3,
    subsample=0.9,
    colsample_bytree=0.5,
    seed=42,
    use_label_encoder=False,
    eval_metric='logloss'  # Avoid warning
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# ----------------------------------------
# 📉 Confusion Matrix
# ----------------------------------------

ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test, display_labels=["Did not leave", "Left"]
)
plt.title("Confusion Matrix")
plt.show()

# ----------------------------------------
# 🧠 Feature Importance (by Gain)
# ----------------------------------------

print("\nTop 10 Feature Importances (by GAIN):")
bst = model.get_booster()
gain_scores = bst.get_score(importance_type='gain')
for feature, score in sorted(gain_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feature}: {score:.4f}")

# Visualize importance
plot_importance(model, max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()

# ----------------------------------------
# 🌳 Visualize First Tree
# ----------------------------------------

plt.figure(figsize=(30, 20))
plot_tree(model, num_trees=0)
plt.title("First Tree in XGBoost Model")
plt.show()

# ----------------------------------------
# 📊 Evaluation Metrics
# ----------------------------------------

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC Curve
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.2f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()