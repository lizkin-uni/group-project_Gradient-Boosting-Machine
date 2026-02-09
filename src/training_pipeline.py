import pandas as pd
from io import StringIO
from google.cloud import storage
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.validation import check_is_fitted

BUCKET_NAME = "gradient_boosting_machine"
TRAIN_PATH = "inputs/train.csv"
VAL_PATH = "inputs/val.csv"
MODEL_OUTPUT_PATH = "pipeline_artifacts/churn_model.pkl"


class ChurnModel(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        self.is_fitted_ = False
        
        num_features = [
            "age",
            "watch_hours",
            "last_login_days",
            "monthly_fee",
            "number_of_profiles",
            "avg_watch_time_per_day"
        ]
        
        cat_features = [
            "gender",
            "subscription_type",
            "region",
            "device",
            "payment_method",
            "favorite_genre"
        ]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ]
        )
        
        self.pipeline_ = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])
    
    def fit(self, X, y):
        self.pipeline_.fit(X, y)
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        return self.pipeline_.predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        return self.pipeline_.predict_proba(X)


def load_csv_from_gcs(bucket_name, file_path):
    print(f"  Loading {file_path} from bucket {bucket_name}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_text()
    df = pd.read_csv(StringIO(data))
    print(f"  Loaded {len(df)} rows")
    return df


def save_model_to_gcs(model, bucket_name, file_path):
    print(f"  Saving model to {file_path}...")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    model_bytes = joblib.dumps(model)
    blob.upload_from_string(model_bytes)
    print(f"  Model saved successfully")


def compute_metrics(name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"\n{name} Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    
    return {"accuracy": acc, "f1_score": f1, "roc_auc": roc_auc}


def main():
    print("=" * 60)
    print("TRAINING PIPELINE - CHURN PREDICTION MODEL")
    print("=" * 60)
    
    print("\n[1/5] Loading training data from Google Cloud Storage...")
    df_train = load_csv_from_gcs(BUCKET_NAME, TRAIN_PATH)
    
    print("\n[2/5] Loading validation data from Google Cloud Storage...")
    df_val = load_csv_from_gcs(BUCKET_NAME, VAL_PATH)
    
    print("\n[3/5] Preparing features...")
    X_train = df_train.drop(columns=["churned", "customer_id"])
    y_train = df_train["churned"]
    
    X_val = df_val.drop(columns=["churned", "customer_id"])
    y_val = df_val["churned"]
    
    print(f"  Train set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    
    print("\n[4/5] Training Gradient Boosting model...")
    model = ChurnModel()
    model.fit(X_train, y_train)
    print("  Training completed")
    
    print("\n[5/5] Evaluating model...")
    
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics("TRAIN", y_train, y_train_pred, y_train_proba)
    
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics("VALIDATION", y_val, y_val_pred, y_val_proba)
    
    print("\n[6/6] Saving model artifact to Google Cloud Storage...")
    save_model_to_gcs(model, BUCKET_NAME, MODEL_OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Model saved to: gs://{BUCKET_NAME}/{MODEL_OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
