import pytest
import pandas as pd
import numpy as np
import warnings
from src.config import QSARConfig
from src.data_acquisition import ChEMBLAcquirer
from src.descriptor_calculator import MolecularDescriptorCalculator
from sklearn.ensemble import RandomForestClassifier

def test_pic50_conversion():
    """Test standard IC50 to pIC50 conversion."""
    config = QSARConfig()
    acquirer = ChEMBLAcquirer(config)
    
    # Mock data: 10 nM and 10,000 nM
    df_mock = pd.DataFrame({
        'molecule_chembl_id': ['CHEMBL1', 'CHEMBL2'],
        'canonical_smiles': ['C', 'CC'],
        'standard_value': [10.0, 10000.0],  # 10nM -> pIC50=8, 10000nM -> pIC50=5
        'standard_units': ['nM', 'nM']
    })
    
    df_processed = acquirer.preprocess_data(df_mock)
    
    # 10nM -> 8 (active, since 8 >= 6)
    # 10000nM -> 5 (inactive, since 5 <= 5)
    assert len(df_processed) == 2
    assert np.isclose(df_processed.iloc[0]['pIC50'], 8.0)
    assert np.isclose(df_processed.iloc[1]['pIC50'], 5.0)
    
    assert df_processed.iloc[0]['activity_label'] == 1
    assert df_processed.iloc[1]['activity_label'] == 0

def test_valid_smiles_parse():
    """Test descriptor calculation with valid and invalid SMILES."""
    config = QSARConfig()
    # Suppress RDKit logging for the invalid SMILES to keep test output clean
    calculator = MolecularDescriptorCalculator(config)
    
    df_mock = pd.DataFrame({
        'canonical_smiles': ['c1ccccc1', 'InvalidSmilesStringHere']
    })
    
    df_processed = calculator.process_dataframe(df_mock)
    
    # Only 1 valid SMILES should survive
    assert len(df_processed) == 1
    assert 'Morgan_0' in df_processed.columns
    assert 'MolWt' in df_processed.columns

def test_model_output_shape():
    """Test model handles structural shape correctly on a mock dataset."""
    X = np.random.rand(10, 20) # 10 samples, 20 features
    y = np.random.randint(0, 2, size=10) # Binary target
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Test prediction on 2 new samples
    X_new = np.random.rand(2, 20)
    preds = model.predict(X_new)
    probs = model.predict_proba(X_new)
    
    assert preds.shape == (2,)
    assert probs.shape == (2, 2)
