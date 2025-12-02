"""
End-to-end ML training pipeline with MLflow tracking.
Loads data from MySQL, trains ensemble model, evaluates, and logs to MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import yaml
import os
from datetime import datetime
import mlflow
from pathlib import Path

from ml.models.ensemble import InsuranceEnsembleModel
from data.scripts.load_raw_data import load_insurance_data_from_csv
from data.schemas.mysql_schema import SCHEMA_SQL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize pipeline with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # MLflow setup
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME', 'insurance_models'))
    
    def load_data(self, csv_path: str = 'Motor vehicle insurance data.csv'):
        """Load and preprocess data from CSV."""
        logger.info(f"Loading data from {csv_path}...")
        
        df = pd.read_csv(csv_path, sep=';', encoding='utf-8', low_memory=False)
        logger.info(f"Loaded dataset shape: {df.shape}")
        
        # Target variable
        y = df[self.config['data']['target']].astype(int)
        
        # Feature selection
        feature_cols = [col for col in df.columns if col != self.config['data']['target'] and col != 'ID']
        X = df[feature_cols].copy()
        
        return X, y, feature_cols
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series, feature_cols: list):
        """Preprocess features."""
        logger.info("Preprocessing data...")
        
        # Handle missing values
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].isnull().sum() > 0:
                if X_processed[col].dtype == 'object':
                    X_processed[col].fillna('Unknown', inplace=True)
                else:
                    X_processed[col].fillna(X_processed[col].median(), inplace=True)
        
        # Date columns
        date_cols = [col for col in X_processed.columns if 'Date' in col or 'date' in col]
        for col in date_cols:
            X_processed[col] = pd.to_datetime(X_processed[col], dayfirst=True, errors='coerce')
            X_processed[col] = X_processed[col].astype(np.int64) // 10**9  # Convert to seconds
        
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        return X_processed
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series):
        """Train ensemble model and evaluate."""
        logger.info("Starting model training...")
        
        with mlflow.start_run():
            # Train-test split
            test_size = self.config['data'].get('test_split', 0.2)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            
            # Validation split
            val_size = self.config['data'].get('validation_split', 0.1)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size / (1 - test_size), 
                stratify=y_temp, random_state=42
            )
            
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Initialize model
            model = InsuranceEnsembleModel(config_path='config.yaml')
            
            # Log parameters
            mlflow.log_params({
                'ensemble_type': 'XGB+LGB+NN',
                'test_size': test_size,
                'validation_size': val_size,
                'random_state': 42
            })
            
            # Train
            logger.info("Training ensemble model...")
            model.train(X_train, y_train, X_val, y_val)
            
            # Evaluate on test set
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            # Check if binary or multiclass
            n_classes = len(np.unique(y_test))
            is_binary = n_classes == 2
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
            
            # ROC-AUC only for binary
            try:
                auc = roc_auc_score(y_test, y_pred_proba) if is_binary else roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                auc = 0.0
            
            logger.info(f"\nModel Performance:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            logger.info(f"  ROC-AUC:   {auc:.4f}")
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': auc
            })
            
            # Cross-validation
            logger.info("\nRunning cross-validation...")
            cv = StratifiedKFold(n_splits=self.config['ml']['validation']['cross_validation_folds'], shuffle=True)
            
            # Use RandomForest if available, else GradientBoosting
            estimator_to_cv = model.rf_model if hasattr(model, 'rf_model') and model.rf_model else model.gb_model
            if estimator_to_cv:
                try:
                    cv_scores = cross_val_score(estimator_to_cv, X_train, y_train, cv=cv, scoring='accuracy')
                    logger.info(f"CV Accuracy scores: {cv_scores}")
                    logger.info(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                    
                    mlflow.log_metrics({
                        'cv_accuracy_mean': cv_scores.mean(),
                        'cv_accuracy_std': cv_scores.std()
                    })
                except Exception as e:
                    logger.warning(f"Could not run cross-validation: {e}")
            else:
                logger.warning("No model available for cross-validation")
            
            # Save model
            models_dir = os.getenv('MODELS_DIR', 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_filename = f"ensemble_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            model_path = os.path.join(models_dir, model_filename)
            
            # Use joblib to serialize the model
            import joblib
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Log model directory to MLflow
            mlflow.log_artifact(models_dir)
            
            return {
                'model': model,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc
                },
                'cv_scores': cv_scores.tolist()
            }
    
    def run(self, csv_path: str = 'Motor vehicle insurance data.csv'):
        """Run complete pipeline."""
        try:
            logger.info("="*60)
            logger.info("Insurance Risk Model Training Pipeline")
            logger.info("="*60)
            
            # Load data
            X, y, feature_cols = self.load_data(csv_path)
            
            # Preprocess
            X_processed = self.preprocess_data(X, y, feature_cols)
            
            # Train and evaluate
            results = self.train_and_evaluate(X_processed, y)
            
            logger.info("\n" + "="*60)
            logger.info("Pipeline completed successfully!")
            logger.info("="*60)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()
