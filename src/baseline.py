import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

class Baseline(BaseEstimator, ClassifierMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.target_mean_ = None

    def fit(self, X, y):
        self.target_mean_ = y.mean()
        self.n_samples_ = len(y)
        return self

    def predict(self, X):
        check_is_fitted(self, "target_mean_")

        rng = np.random.default_rng(self.random_state)
        return rng.binomial(1, self.target_mean_, size=X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, "target_mean_")

        p = self.target_mean_
        proba = np.column_stack([np.full(X.shape[0], 1 - p), np.full(X.shape[0], p)])
        return proba

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os

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

baseline = Baseline(random_state=42)
baseline.fit(X_train, y_train)

def _print_metrics(name, y_true, y_pred, y_proba):
    print(f"{name} - Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    try:
        print(f"{name} - ROC-AUC: {roc_auc_score(y_true, y_proba):.4f}")
    except ValueError:
        print(f"{name} - ROC-AUC: N/A")
    print(f"{name} - F1-Score: {f1_score(y_true, y_pred):.4f}")

# Train metrics
y_train_pred = baseline.predict(X_train)
y_train_proba = baseline.predict_proba(X_train)[:, 1]
_print_metrics("Baseline_Metrics_Train", y_train, y_train_pred, y_train_proba)

# Validation metrics
y_val_pred = baseline.predict(X_val)
y_val_proba = baseline.predict_proba(X_val)[:, 1]
_print_metrics("Baseline_Metrics_Validation", y_val, y_val_pred, y_val_proba)