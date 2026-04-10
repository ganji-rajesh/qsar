import joblib
import pandas as pd
import logging
import os
from typing import Dict, Any
from .descriptor_calculator import MolecularDescriptorCalculator

class QSARPredictor:
    """Loads a trained model and config to predict activity for new SMILES."""
    
    def __init__(self, model_path: str, config_path: str):
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError("Model or Config file not found. Please train the model first.")
            
        logging.info("Loading linked configuration...")
        self.config = joblib.load(config_path)
        
        logging.info("Loading model...")
        self.model = joblib.load(model_path)
        
        # Initialize the calculator exactly as it was during training
        self.calculator = MolecularDescriptorCalculator(self.config)
        
        # Determine expected feature columns from config/calculator setup
        fp_cols = [f"Morgan_{i}" for i in range(self.config.morgan_nbits)]
        phys_cols = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors']
        self.feature_cols = fp_cols + phys_cols

    def predict(self, smiles: str) -> Dict[str, Any]:
        """Predict activity for a single SMILES string."""
        # Calculate descriptors
        df = self.calculator.process_smiles_list([smiles])
        
        if df.empty or self.feature_cols[0] not in df.columns:
            return {"error": "Invalid SMILES structure or descriptor calculation failed."}
            
        X = df[self.feature_cols]
        
        prediction = self.model.predict(X)[0]
        result = {"prediction": int(prediction), "smiles": smiles}
        
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(X)[0]
            result["probability_active"] = float(prob[1])
            
        return result
