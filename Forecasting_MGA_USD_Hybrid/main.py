import config
from src.Arima_feature_selection import select_arima_lags
from src.Data_preparation import prepare_supervised_data
from src.Model_training import train_and_save_models
from src.walk_forward_validation import evaluate_predictions
from src.Financial_metrics import compute_financial_metrics

def main():
    print("=== DÉBUT DU PIPELINE DE PRÉVISION HYBRIDE USD/MGA ===")

    # ÉTAPE 1 : Sélection des retards (Features)
    print("\n--- 1. SÉLECTION DES FEATURES (ARIMA) ---")
    best_lags = select_arima_lags(data_path=config.DATA_RAW, max_lags=config.MAX_LAGS)
    print(f"=> Lags optimaux retenus : {best_lags}")

    # ÉTAPE 2 : Préparation et sauvegarde des données
    print("\n--- 2. PRÉPARATION DES DONNÉES ---")
    data = prepare_supervised_data(
        data_path=config.DATA_RAW,
        lags=best_lags,
        test_size=config.TEST_SIZE,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    dates_test = data["dates_test"]

    # Transformation de Quant : Travailler sur les variations (Delta y)
    y_train_diff = data["y_train_diff"] 
    y_test_lag1 = data["y_test_lag1"]

    # ÉTAPE 3 : Entraînement et sauvegarde des modèles (.pkl)
    print("\n--- 3. ENTRAÎNEMENT ET SAUVEGARDE DES MODÈLES ---")
    trained_models = train_and_save_models(X_train, y_train_diff)

    # ÉTAPE 4 : Validation (Prédictions sur X_test)
    print("\n--- 4. ÉVALUATION ET WALK-FORWARD VALIDATION ---")
    predictions = evaluate_predictions(trained_models, X_test, y_test_lag1)

    # ÉTAPE 5 : Métriques Financières (Sharpe, Hit Rate)
    print("\n--- 5. CALCUL DES MÉTRIQUES FINANCIÈRES ---")
    compute_financial_metrics(y_test, predictions, dates_test)

    print("\n=== PIPELINE TERMINÉ AVEC SUCCÈS ! ===")

if __name__ == "__main__":
    main()