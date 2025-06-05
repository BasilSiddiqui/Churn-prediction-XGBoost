# 📊 Telco Customer Churn Prediction using XGBoost

This project aims to predict customer churn for a telecom company using the **XGBoost** classification algorithm. We built, trained, and evaluated a model using real-world customer data.

---

## 📁 Dataset

The dataset used in this project is [`Telco-Customer-Churn.csv`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). It contains customer information such as services subscribed, tenure, monthly charges, contract types, and more.

**Target Variable:**
- `Churn`: Indicates whether a customer left (Yes) or stayed (No).

---

## 🧼 Data Preprocessing

1. **Removed Irrelevant Features**:
   - Dropped the `customerID` column as it’s just an identifier.

2. **Handled Missing Data**:
   - Replaced blank `" "` entries in `TotalCharges` with `0`, then converted the column to numeric.

3. **Reformatted Column Values**:
   - Replaced whitespace in strings with underscores for consistent formatting.

4. **Converted Categorical Variables**:
   - Applied one-hot encoding to all categorical features, including:
     - `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`,
     - `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
     - `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`,
     - `PaperlessBilling`, `PaymentMethod`

5. **Mapped Target Labels**:
   - Converted `Churn` values from `Yes/No` to `1/0`.

---

## 🧪 Model Training

We used the following **XGBoost** classifier settings:

```python
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
````

The dataset was split 80/20 into training and testing sets using `train_test_split` with stratification on the target variable.

The model was trained with an evaluation set:

```python
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
```

---

## 📉 Evaluation

A **confusion matrix** was used to visualize the model's classification performance.

📌 *Insert Confusion Matrix here:*

![Confusion Matrix](images/ConfusionMatrix.png)

---

## 🔍 Feature Importance

We extracted feature importances from the trained XGBoost model using the **`gain`** metric:

```python
bst = model.get_booster()
gain_scores = bst.get_score(importance_type='gain')
```

The top 10 features based on average gain are printed in the console.

---

## 🌳 Tree Visualization

We visualized the **first decision tree** in the ensemble using XGBoost’s `plot_tree()` function. The visualization uses custom styling for internal and leaf nodes for improved clarity.

📌 *Insert Tree Visualization here:*

![XGBoost Tree](images/TreeVisualization.png)

---

## 💡 Future Improvements

* Implement early stopping for faster training (requires version-specific compatibility).
* Explore hyperparameter tuning with `GridSearchCV`.
* Add ROC-AUC, PR-AUC, and feature selection for enhanced evaluation.

---

## ⚙️ Requirements

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* xgboost (v3.0.2)

To install dependencies via conda:

```bash
conda install -c conda-forge xgboost scikit-learn matplotlib pandas numpy
```

---

## 📂 File Structure

```
├── Telco-Customer-Churn.csv
├── churn_model.py                # Main model training and evaluation script
├── images/
│   ├── confusion_matrix.png
│   └── tree_visualization.png
└── README.md
```

---

## 👤 Author

**Basil Rehan**
Data Analyst | Actuarial & Data Science Student
Heriot-Watt University Dubai

```

---

Let me know if you want me to generate the images or the `.py` file as well!
```
