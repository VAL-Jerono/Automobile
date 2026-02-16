"""
MLflow tracking integration
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MLflowTracker:
    """MLflow experiment tracking wrapper"""
    
    def __init__(self, tracking_uri: str = "file:./mlruns", experiment_name: str = "insurance_analytics"):
        """
        Initialize MLflow tracker
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of MLflow experiment
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracker initialized: {experiment_name}")
    
    def log_model_training(self, model_name: str, model, metrics: dict, params: dict):
        """
        Log model training run
        
        Args:
            model_name: Name of the model
            model: Trained model object
            metrics: Dictionary of metrics
            params: Dictionary of parameters
        """
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Log tags
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("training_date", datetime.now().isoformat())
            
            logger.info(f"Logged training run for {model_name}")
    
    def log_prediction_batch(self, model_name: str, batch_size: int, avg_confidence: float):
        """
        Log prediction batch metrics
        
        Args:
            model_name: Name of the model
            batch_size: Number of predictions
            avg_confidence: Average prediction confidence
        """
        with mlflow.start_run(run_name=f"{model_name}_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metric("batch_size", batch_size)
            mlflow.log_metric("avg_confidence", avg_confidence)
            mlflow.set_tag("prediction_timestamp", datetime.now().isoformat())
            
            logger.info(f"Logged prediction batch for {model_name}: {batch_size} predictions")
    
    def get_latest_model(self, model_name: str):
        """
        Get latest version of a model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model
        """
        try:
            model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
            logger.info(f"Loaded latest model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None
