"""
Pipeline principal avec Walk-Forward Séquentiel.
À exécuter après avoir vérifié que l'ancien main.py fonctionnait.
"""

import config
from src.Arima_feature_selection import select_arima_lags
from src.Data_preparation import prepare_supervised_data
from src.Model_training import train_and_save_models
from src.walk_forward_sequential import walk_forward_predict
from src.Financial_metrics import compute_financial_metrics


def main():
    print("=" * 60)
    print(" PIPELINE WALK-FORWARD SÉQUENTIEL — USD/MGA")
    print("=" * 60)

    # ÉTAPE 1 : Sélection des features (ARIMA)
    print("\n--- 1. SÉLECTION DES FEATURES ---")
    best_lags = select_arima_lags(data_path=config.DATA_RAW, max_lags=config.MAX_LAGS)
    print(f"=> Lags optimaux retenus : {best_lags}")

    # ÉTAPE 2 : Préparation des données
    print("\n--- 2. PRÉPARATION DES DONNÉES ---")
    data = prepare_supervised_data(
        data_path=config.DATA_RAW,
        lags=best_lags,
        test_size=config.TEST_SIZE,
    )
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    dates_test = data["dates_test"]
    y_train_diff = data["y_train_diff"]
    y_test_lag1 = data["y_test_lag1"]

    # ÉTAPE 3 : Entraînement initial
    print("\n--- 3. ENTRAÎNEMENT INITIAL ---")
    trained_models = train_and_save_models(X_train, y_train_diff)
    print(f"=> Modèles entraînés : {list(trained_models.keys())}")

    # ÉTAPE 4 : Walk-Forward séquentiel
    print("\n--- 4. WALK-FORWARD SÉQUENTIEL ---")
    # Configuration du réentraînement :
    # - XGBoost : rapide, on peut réentraîner périodiquement
    # - SVR / MLP / LSTM : trop lents ou pas de partial_fit, on garde le modèle initial
    retrain_config = {
        "XGBoost": True,
        "SVR": False,
        "MLP": False,
        "LSTM": False,
    }

    results_df = walk_forward_predict(
        trained_models=trained_models,
        X_test=X_test,
        y_test_lag1=y_test_lag1,
        y_test=y_test,
        retrain_models=retrain_config,
        retrain_freq=60,           # Réentraîner XGBoost tous les 60 jours
        X_train=X_train,
        y_train_diff=y_train_diff,
        lstm_names=["LSTM"]        # Seul le LSTM nécessite reshape 3D
    )

    # Sauvegarde
    out_file = "results_walk_forward.csv"
    results_df.to_csv(out_file)
    print(f"\n=> Résultats sauvegardés dans : {out_file}")

    # ÉTAPE 5 : Métriques financières (Sharpe, etc.)
    print("\n--- 5. MÉTRIQUES FINANCIÈRES ---")
    # Conversion au format attendu par compute_financial_metrics
    predictions_dict = {col: results_df[col] for col in results_df.columns}
    compute_financial_metrics(y_test, predictions_dict, dates_test)

    print("\n" + "=" * 60)
    print(" PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 60)


if __name__ == "__main__":
    main()
