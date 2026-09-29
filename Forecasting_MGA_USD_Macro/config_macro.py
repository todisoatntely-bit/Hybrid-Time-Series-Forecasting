import os
from pathlib import Path

# --- CHEMINS (PATHS) ---
BASE_DIR = Path(__file__).resolve().parent

# Data
DATA_RAW_PANEL = BASE_DIR / "data" / "raw" / "panel_mensuel_mga.csv"
DATA_PROCESSED_TRAIN = BASE_DIR / "data" / "processed" / "macro_train.csv"
DATA_PROCESSED_TEST = BASE_DIR / "data" / "processed" / "macro_test.csv"

# Models & Outputs
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

# Figures SHAP & Prévisions
SHAP_BAR_PLOT = FIGURES_DIR / "shap_bar_plot.png"
SHAP_BEESWARM_PLOT = FIGURES_DIR / "shap_beeswarm_plot.png"
MACRO_FORECAST_PLOT = FIGURES_DIR / "macro_forecast_plot.png"

# Création automatique des dossiers
for dir_path in [MODELS_DIR, FIGURES_DIR, TABLES_DIR, BASE_DIR / "data" / "processed"]:
    os.makedirs(dir_path, exist_ok=True)

# --- PARAMÈTRES GLOBAUX ---
TEST_SIZE = 12
TARGET_VAR = "mga_usd"