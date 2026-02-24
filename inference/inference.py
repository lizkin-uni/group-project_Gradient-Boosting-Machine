import pandas as pd
import joblib
from google.cloud import storage
import io
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def run_inference(bucket_name, model_path, input_path, output_path):
    try:
        client = storage.Client(project='excercise-1-485615')
        bucket = client.bucket(bucket_name)
        
        logger.info(f"Downloading model from gs://{bucket_name}/{model_path}...")
        model_blob = bucket.blob(model_path)
        model_buffer = io.BytesIO()
        model_blob.download_to_file(model_buffer)
        model_buffer.seek(0)
        model = joblib.load(model_buffer)
        logger.info("Model loaded successfully")
        
        logger.info(f"Reading test data from gs://{bucket_name}/{input_path}...")
        input_blob = bucket.blob(input_path)
        data = input_blob.download_as_text()
        test_df = pd.read_csv(io.StringIO(data))
        logger.info(f"Loaded {len(test_df)} rows of test data")
        
        logger.info("Running predictions...")
        predictions = model.predict(test_df)
        
        output_df = pd.DataFrame({'churn_prediction': predictions})
        
        logger.info(f"Uploading results to gs://{bucket_name}/{output_path}...")
        output_blob = bucket.blob(output_path)
        output_blob.upload_from_string(output_df.to_csv(index=False), 'text/csv')
        
        logger.info(f"Inference pipeline completed successfully! Generated {len(output_df)} predictions.")
        
    except Exception as e:
        logger.error(f"Error during inference pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    BUCKET = "gradient_boosting_machine"
    
    run_inference(
        bucket_name=BUCKET,
        model_path="pipeline_artifacts/pipeline_churn_model.pkl",
        input_path="inputs/test.csv",
        output_path="outputs/predictions.csv"
    )