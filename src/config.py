from dataclasses import dataclass
from typing import Tuple

@dataclass
class QSARConfig:
    """Centralized configuration for the QSAR Model pipeline."""
    # Data Acquisition Parameters
    target_chembl_id: str = 'CHEMBL203'  # Default Target: EGFR
    max_records: int = 1000              # Limit download records (set to None for all)
    active_threshold: float = 6.0        # pIC50 >= 6.0 is active
    inactive_threshold: float = 5.0      # pIC50 <= 5.0 is inactive
    
    # Descriptor Calculation Parameters
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    
    # Model Training Parameters
    test_size: float = 0.2
    random_state: int = 42
    n_estimators_rf: int = 100
    
    # Folders
    data_dir: str = 'data'
    output_dir: str = 'outputs'
    
    @property
    def model_path(self) -> str:
        return f"{self.data_dir}/qsar_model.pkl"
        
    @property
    def config_path(self) -> str:
        return f"{self.data_dir}/qsar_config.pkl"
