# QSAR Model Implementation Plan

This implementation plan details the architecture and workflow for building the QSAR model to predict molecular toxicity/activity based on the required phases outlined.

## Proposed Changes

We will adopt an Object-Oriented design, separating concerns into logical modules. All code will be developed in `d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR`.

### Core Pipeline Modules

#### [NEW] [data_acquisition.py](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/src/data_acquisition.py)
Class `ChEMBLAcquirer` to interact with `chembl_webresource_client`. Features robust error handling, target query, and basic filtering (converting IC50 to pIC50, thresholding, handling missing values).

#### [NEW] [descriptor_calculator.py](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/src/descriptor_calculator.py)
Class `MolecularDescriptorCalculator` to parse SMILES with `rdkit`. Will validate structures, generate Morgan/MACCS/RDKit fingerprints, and compute physicochemical properties (MW, LogP).

#### [NEW] [model_trainer.py](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/src/model_trainer.py)
Class `QSARTrainer` to handle scikit-learn models (Random Forest, SVM). Will manage train-test splits, feature scaling, hyperparameter tuning (`GridSearchCV`), and model evaluation (accuracy, precision, recall, F1, ROC-AUC, MCC). Generates feature importance and relevant plots.

#### [NEW] [model_deployer.py](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/src/model_deployer.py)
Class `QSARPredictor` to load trained models via `joblib`, process new SMILES strings through the same descriptor pipeline, and provide predictions.

#### [NEW] [main.py](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/main.py)
The primary entry point that coordinates the classes, provides progress tracking (`tqdm`), and ensures intermediate results and PNG visualizations are saved in an `outputs/` and `data/` directory.

### Documentation and Configurations

#### [NEW] [README.md](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/README.md)
Comprehensive markdown file including Project Overview, Installation Instructions, Usage Guide, Architecture Diagram, Results Section, Dependencies, Troubleshooting, and References.

#### [NEW] [requirements.txt](file:///d:/G_RAJESH/UOH_MBA/projects2.0/cheminfomatics/QSAR/requirements.txt)
List of required dependencies: `chembl_webresource_client`, `rdkit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `tqdm`.

## User Review Required

> [!WARNING]
> This plan sets up the complete scaffolding and logic for the QSAR ML system. The user defined the exact scope. Could you please confirm if this class structure and file separation meet your portfolio demonstration expectations? Also, please provide a preferred default target protein (e.g., 'CHEMBL203', which is EGFR, or 'CHEMBL244' which is AChE) to use for the end-to-end default run, or I will use EGFR as the default.

## Verification Plan

### Automated / Manual Verification
1. Install dependencies per `requirements.txt`.
2. Run `main.py` directly. Verify that it executes from start to finish without crashing.
3. Check the `outputs/` directory for generated plots, models, and summary metrics.
4. Call `predict` functions via a small test script to verify new compounds map correctly to the active/inactive space.
