import config_macro
from src.step01_macro_data_preparation import prepare_macro_data
from src.step02_model_training import train_macro_models
from src.step03_evaluation import evaluate_macro_models
from src.step04_economic_interpretation import interpret_economic_drivers

def main():
    print("=== DÉBUT DU PIPELINE MACROÉCONOMIQUE (VOLET 2) ===")

    # ÉTAPE 1 : Préparation des données (Imputation IPC & Lags)
    print("\n--- 1. PRÉPARATION DES DONNÉES MACROÉCONOMIQUES ---")
    data = prepare_macro_data(
        data_path=config_macro.DATA_RAW_PANEL,
        test_size=config_macro.TEST_SIZE
    )
    print("=> Données préparées et variables décalées (t-1) avec succès.")
    
    # ÉTAPE 2 : Entraînement des modèles sur les fondamentaux
    print("\n--- 2. ENTRAÎNEMENT DES MODÈLES ---")
    trained_models = train_macro_models(data["X_train"], data["y_train_diff"])

    # ÉTAPE 3 : Évaluation des performances
    print("\n--- 3. ÉVALUATION DES PRÉDICTIONS ---")
    evaluate_macro_models(
        trained_models, 
        data["X_test"], 
        data["y_test_brute"], 
        data["y_test_lag1"],
        data["dates_test"]
    )

    # ÉTAPE 4 : Interprétation Économique (SHAP) - La clé du mémoire
    print("\n--- 4. INTERPRÉTATION ÉCONOMIQUE (SHAP) ---")
    interpret_economic_drivers(
        trained_models, 
        data["X_train"], 
        data["feature_names"]
    )

    print("\n=== PIPELINE TERMINÉ AVEC SUCCÈS ! ===")

if __name__ == "__main__":
    main()