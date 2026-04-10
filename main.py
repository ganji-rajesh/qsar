import logging
import os
import json
from src.config import QSARConfig
from src.data_acquisition import ChEMBLAcquirer
from src.descriptor_calculator import MolecularDescriptorCalculator
from src.model_trainer import QSARTrainer
from src.model_deployer import QSARPredictor

def run_pipeline():
    logging.info("Starting QSAR Pipeline...")
    
    # 1. Configuration
    config = QSARConfig()
    
    # Create required directories before saving files
    os.makedirs(config.data_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 2. Data Acquisition
    logging.info("--- Phase 1: Data Acquisition ---")
    acquirer = ChEMBLAcquirer(config)
    df_raw = acquirer.run_pipeline()
    
    if df_raw.empty:
        logging.error("No data acquired. Exiting pipeline.")
        return
        
    # 3. Descriptor Calculation
    logging.info("--- Phase 2: Descriptor Calculation ---")
    calculator = MolecularDescriptorCalculator(config)
    df_features = calculator.process_dataframe(df_raw)
    
    # Save processed dataframe
    features_path = os.path.join(config.data_dir, 'processed_features.csv')
    df_features.to_csv(features_path, index=False)
    logging.info(f"Saved preprocessed data to {features_path}")
    
    # 4. Model Training
    logging.info("--- Phase 3 & 4: Model Training and Optimization ---")
    trainer = QSARTrainer(config, df_features)
    trainer.prepare_data()
    
    rf_model = trainer.train_random_forest()
    svm_model = trainer.train_svm()
    
    # 5. Model Evaluation
    logging.info("--- Phase 5: Model Evaluation ---")
    rf_metrics = trainer.evaluate_model(rf_model, "Random Forest")
    svm_metrics = trainer.evaluate_model(svm_model, "SVM")
    
    trainer.plot_feature_importance()
    
    # Select best model (simplified here to always be RF or SVM based on ROC_AUC or Accuracy)
    rf_score = rf_metrics.get('ROC_AUC', rf_metrics['Accuracy'])
    svm_score = svm_metrics.get('ROC_AUC', svm_metrics['Accuracy'])
    
    best_model = rf_model if rf_score >= svm_score else svm_model
    best_name = "Random Forest" if rf_score >= svm_score else "SVM"
    logging.info(f"Selected best model: {best_name}")
    
    # 6. Deployment / Serialization
    logging.info("--- Phase 6: Model Deployment ---")
    trainer.save_artifacts(best_model)
    
    # Generate final report
    report = {
        "Target": config.target_chembl_id,
        "Total Compounds Processed": len(df_features),
        "Random Forest Metrics": rf_metrics,
        "SVM Metrics": svm_metrics,
        "Selected Model": best_name
    }
    with open(os.path.join(config.output_dir, "summary_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info("Pipeline completed successfully!")
    
    # Quick Test of Predictor
    logging.info("--- Testing Deployer ---")
    predictor = QSARPredictor(config.model_path, config.config_path)
    # Using Gefitinib (an EGFR inhibitor) SMILES as test
    test_smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4"
    res = predictor.predict(test_smiles)
    logging.info(f"Test Prediction for Gefitinib: {res}")


if __name__ == "__main__":
    run_pipeline()
