import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

# ----------------------------------------
# 📥 Load and Clean Data
# ----------------------------------------

# Load dataset
df = pd.read_csv(r'C:\Users\basil\OneDrive\Desktop\Base\Work\Personal projects\Telco-Customer-Churn.csv')

# Drop customerID as it's not useful for modeling
df.drop(['customerID'], axis=1, inplace=True)

# Handle empty values in TotalCharges
df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# Replace spaces with underscores for consistency
df = df.replace(' ', '_', regex=True)

# ----------------------------------------
# 🎯 Feature and Target Setup
# ----------------------------------------

# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert to binary labels

# One-hot encode categorical variables
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
    use_label_encoder=False
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# ----------------------------------------
# 📉 Confusion Matrix
# ----------------------------------------

ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test,
    display_labels=["Did not leave", "Left"]
)
plt.title("Confusion Matrix")
plt.show()

# ----------------------------------------
# 🔍 Feature Importance
# ----------------------------------------

print("\nTop 10 Feature Importances (by GAIN):")
bst = model.get_booster()
gain_scores = bst.get_score(importance_type='gain')
for feature, score in sorted(gain_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feature}: {score:.4f}")

# ----------------------------------------
# 🌳 Visualize First Tree
# ----------------------------------------

node_params = {
    'shape': 'box',
    'style': 'filled, rounded',
    'fillcolor': '#78cbe0'
}

leaf_params = {
    'shape': 'box',
    'style': 'filled',
    'fillcolor': '#e48038'
}

plt.figure(figsize=(30, 20))
plot_tree(
    model,
    num_trees=0,  # Visualize the first tree
    rankdir='LR',
    condition_node_params=node_params,
    leaf_node_params=leaf_params
)
plt.title("First Tree in XGBoost Model")
plt.show()