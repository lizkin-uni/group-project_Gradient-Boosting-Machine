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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

baseline = Baseline(random_state=42)
baseline.fit(X_train, y_train)

y_pred = baseline.predict(X_test)
y_proba = baseline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("F1-Score:", f1_score(y_test, y_pred))