import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from .config import QSARConfig
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ChEMBLAcquirer:
    """Class to interact with ChEMBL API and preprocess bioactivity data."""
    
    def __init__(self, config: QSARConfig):
        self.config = config
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch IC50 bioactivity data for the target from ChEMBL."""
        logging.info(f"Querying ChEMBL for target: {self.config.target_chembl_id}")
        
        try:
            from tqdm import tqdm
            activity = new_client.activity
            res = activity.filter(
                target_chembl_id=self.config.target_chembl_id,
                standard_type="IC50",
                relation="=",
                assay_type="B"
            ).only(['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units'])
            
            total_records = len(res)
            logging.info(f"Target contains {total_records} records. Beginning download...")
            
            data = []
            max_limit = self.config.max_records
            display_total = min(total_records, max_limit) if max_limit else total_records
            
            for item in tqdm(res, total=display_total, desc="Fetching Data"):
                data.append(item)
                if max_limit and len(data) >= max_limit:
                    break
            
            if not data:
                logging.warning("No data found for this target with the given constraints.")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            logging.info(f"Retrieved {len(df)} initial records.")
            return df
        except Exception as e:
            logging.error(f"Error fetching data from ChEMBL: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert IC50 to pIC50, adding classification labels."""
        if df.empty:
            return df
            
        # Keep only essential columns and remove missing SMILES or standard values
        essential_cols = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'standard_units']
        available_cols = [col for col in essential_cols if col in df.columns]
        df = df[available_cols].copy()
        df.dropna(subset=['canonical_smiles', 'standard_value'], inplace=True)
        
        # Ensure standard_value is numeric
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df.dropna(subset=['standard_value'], inplace=True)
        
        # Remove duplicates
        df.drop_duplicates(subset=['molecule_chembl_id'], inplace=True)
        
        # Convert IC50 (nM) to pIC50
        # ChEMBL standard_value for IC50 is typically in nM
        # pIC50 = -log10(IC50 * 10^-9) = 9 - log10(IC50)
        def to_pic50(value):
            if value > 0:
                pIC50 = 9 - np.log10(value)
                # Cap negative values or exceptionally high values 
                return max(0, min(15, pIC50))
            return 0
            
        df['pIC50'] = df['standard_value'].apply(to_pic50)
        
        # Classification
        def classify(pic50):
            if pic50 >= self.config.active_threshold:
                return 1
            elif pic50 <= self.config.inactive_threshold:
                return 0
            else:
                return np.nan # Grey-zone
                
        df['activity_label'] = df['pIC50'].apply(classify)
        
        # Log and drop grey-zone compounds
        initial_len = len(df)
        grey_zone = df['activity_label'].isna().sum()
        df.dropna(subset=['activity_label'], inplace=True)
        df['activity_label'] = df['activity_label'].astype(int)
        
        logging.info(f"Preprocessing dropped {grey_zone} grey-zone compounds ({self.config.inactive_threshold} < pIC50 < {self.config.active_threshold}).")
        logging.info(f"Final preprocessed dataset contains {len(df)} compounds.")
        
        return df

    def run_pipeline(self) -> pd.DataFrame:
        """Fetch and preprocess data end-to-end."""
        df_raw = self.fetch_data()
        df_clean = self.preprocess_data(df_raw)
        return df_clean
