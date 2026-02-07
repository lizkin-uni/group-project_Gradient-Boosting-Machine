import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

class Baseline(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y):
        values, counts = np.unique(y, return_counts=True)
        self.majority_class_ = values[np.argmax(counts)]
        self.class_probs_ = counts / counts.sum()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        return np.full(len(X), self.majority_class_)

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        return np.tile(self.class_probs_, (len(X), 1))


data_path = os.path.join(os.path.dirname(__file__), "../input/WatchAlways_customer_churn.csv")
df = pd.read_csv(data_path)

X = df.drop(columns="churned")
y = df["churned"]

# Train (70%), validation (15%) and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

baseline = Baseline()
baseline.fit(X_train, y_train)

def print_metrics(name, y_true, y_pred, y_proba):
    print(f"{name} - Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"{name} - F1-Score: {f1_score(y_true, y_pred):.4f}")
    try:
        print(f"{name} - ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    except ValueError:
        print(f"{name} - ROC-AUC: N/A")


# Train metrics
y_train_pred = baseline.predict(X_train)
y_train_proba = baseline.predict_proba(X_train)[:, 1]
print_metrics("Baseline_Metrics_Train", y_train, y_train_pred, y_train_proba)

# Validation metrics
y_val_pred = baseline.predict(X_val)
y_val_proba = baseline.predict_proba(X_val)[:, 1]
print_metrics("Baseline_Metrics_Validation", y_val, y_val_pred, y_val_proba)

#combine x and y for saving
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

#convert the data to csv
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)
