import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from .config import QSARConfig
import logging

class MolecularDescriptorCalculator:
    """Calculates molecular footprints and descriptors from SMILES."""
    
    def __init__(self, config: QSARConfig):
        self.config = config
        
    def _validate_and_parse(self, smiles: str):
        """Parse SMILES string into an RDKit Mol object, returns None if invalid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Basic standardization validation
                Chem.SanitizeMol(mol)
                return mol
        except Exception:
            return None
        return None

    def _get_morgan_fingerprint(self, mol) -> np.ndarray:
        """Calculate Morgan fingerprint as numpy array."""
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=self.config.morgan_radius, 
            nBits=self.config.morgan_nbits
        )
        arr = np.zeros((1,), dtype=int)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _get_physicochemical_descriptors(self, mol) -> dict:
        """Calculate MW, LogP, HBD, HBA."""
        return {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol)
        }

    def process_dataframe(self, df: pd.DataFrame, smiles_col: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Process a DataFrame containing SMILES strings.
        Returns a new DataFrame with calculated descriptors and valid molecules only.
        """
        data = []
        valid_indices = []
        
        logging.info("Calculating descriptors for molecules...")
        for idx, row in df.iterrows():
            mol = self._validate_and_parse(row[smiles_col])
            
            if mol is not None:
                # Get fingerprints
                morgan_fp = self._get_morgan_fingerprint(mol)
                
                # Get physicochemical descriptors
                phys_desc = self._get_physicochemical_descriptors(mol)
                
                # Combine
                features = {f"Morgan_{i}": val for i, val in enumerate(morgan_fp)}
                features.update(phys_desc)
                
                data.append(features)
                valid_indices.append(idx)
                
        features_df = pd.DataFrame(data)
        features_df.index = valid_indices
        
        # Validate count
        invalid_count = len(df) - len(valid_indices)
        if invalid_count > 0:
            logging.warning(f"Failed to process {invalid_count} invalid SMILES strings.")
            
        # Combine original df with features (for valid rows)
        result_df = pd.concat([df.loc[valid_indices], features_df], axis=1)
        return result_df
        
    def process_smiles_list(self, smiles_list: list) -> pd.DataFrame:
        """Process a raw list of SMILES without an existing DataFrame context."""
        df = pd.DataFrame({'canonical_smiles': smiles_list})
        return self.process_dataframe(df)
