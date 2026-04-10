import pandas as pd
import numpy as np
import os
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from .config import QSARConfig

class QSARTrainer:
    """Trains, evaluates, and serializes QSAR machine learning models."""
    
    def __init__(self, config: QSARConfig, df: pd.DataFrame):
        self.config = config
        self.df = df
        
        # Ensure directories exist
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.feature_cols = [c for c in df.columns if c.startswith('Morgan_') or c in ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']]
        self.target_col = 'activity_label'
        
    def prepare_data(self):
        """Split data and handle scaling as part of the pipeline later."""
        X = self.df[self.feature_cols]
        y = self.df[self.target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        logging.info(f"Data split: Train={self.X_train.shape}, Test={self.X_test.shape}")
        
    def train_random_forest(self):
        """Train and optimize Random Forest model."""
        logging.info("Training Random Forest...")
        
        # In a real scenario, use GridSearchCV, but we'll use a fixed set for speed in portfolio demo
        # while demonstrating the setup.
        rf = RandomForestClassifier(
            n_estimators=self.config.n_estimators_rf,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.rf_model = rf
        return rf
        
    def train_svm(self):
        """Train SVM model with scaling."""
        logging.info("Training SVM...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=self.config.random_state))
        ])
        
        pipeline.fit(self.X_train, self.y_train)
        self.svm_model = pipeline
        return pipeline
        
    def evaluate_model(self, model, model_name: str):
        """Evaluate model and save visualizations."""
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1': f1_score(self.y_test, y_pred),
            'MCC': matthews_corrcoef(self.y_test, y_pred)
        }
        
        if y_prob is not None:
            metrics['ROC_AUC'] = roc_auc_score(self.y_test, y_prob)
            
        logging.info(f"{model_name} Metrics: {metrics}")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/{model_name.lower().replace(' ', '_')}_cm.png")
        plt.close()
        
        return metrics
        
    def plot_feature_importance(self):
        """Plot Random Forest feature importance."""
        if not hasattr(self, 'rf_model'):
            return
            
        importances = self.rf_model.feature_importances_
        # Get top 20
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(10, 6))
        plt.title("Top 20 Feature Importances (Random Forest)")
        plt.bar(range(20), importances[indices], align="center")
        plt.xticks(range(20), [self.feature_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{self.config.output_dir}/rf_feature_importance.png")
        plt.close()
        
    def save_artifacts(self, best_model):
        """Serialize the best model and the configuration."""
        logging.info(f"Saving model to {self.config.model_path}")
        joblib.dump(best_model, self.config.model_path)
        
        logging.info(f"Saving linked config to {self.config.config_path}")
        joblib.dump(self.config, self.config.config_path)
