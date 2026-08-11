import os
from pathlib import Path

# --- CHEMINS (PATHS) ---
BASE_DIR = Path(__file__).resolve().parent

# Data
DATA_RAW = BASE_DIR / "data" / "raw" / "mga_usd_daily_ouvres.csv"
DATA_PROCESSED_TRAIN = BASE_DIR / "data" / "processed" / "train.csv"
DATA_PROCESSED_TEST = BASE_DIR / "data" / "processed" / "test.csv"

# Models & Outputs
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

# Création automatique des dossiers s'ils n'existent pas
for dir_path in [MODELS_DIR, FIGURES_DIR, TABLES_DIR, BASE_DIR / "data" / "processed"]:
    os.makedirs(dir_path, exist_ok=True)

# --- PARAMÈTRES GLOBAUX ---
MAX_LAGS = 10
TEST_SIZE = 60
SIGNIFICANCE_LEVEL = 0.05