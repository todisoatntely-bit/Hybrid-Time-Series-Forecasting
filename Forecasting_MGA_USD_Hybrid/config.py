import os
from pathlib import Path

# ============================================================
# 1. ARBORESCENCE ET CHEMINS (PATHS)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Data
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATA_RAW = DATA_RAW_DIR / "mga_usd_daily_ouvres.csv"
DATA_PROCESSED_TRAIN = DATA_PROCESSED_DIR / "train.csv"
DATA_PROCESSED_TEST = DATA_PROCESSED_DIR / "test.csv"

# Models & Outputs
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Création automatique de toute la structure de dossiers
ALL_DIRS = [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    RESULTS_DIR
]

for dir_path in ALL_DIRS:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. PARAMÈTRES GLOBAUX DU PIPELINE (HYPERPARAMÈTRES)
# ============================================================
MAX_LAGS = 10           # Nombre maximal de retards autorisés pour l'ARIMA
TEST_SIZE = 252         # Taille du jeu de test (ex: 252 jours = 1 an de trading)
SIGNIFICANCE_LEVEL = 0.05