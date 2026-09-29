import pandas as pd
import config
from src.Arima_feature_selection import select_arima_lags
from src.Data_preparation import prepare_supervised_data
from src.Model_training import train_and_save_models
from src.walk_forward_sequential import walk_forward_predict
from src.Financial_metrics import compute_financial_metrics_validated

def main():
    print("=" * 60)
    print(" PIPELINE WALK-FORWARD SÉQUENTIEL — USD/MGA (VOLET 1)")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. PRÉ-DÉCOUPAGE TEMPOREL POUR ÉVITER LE DATA LEAKAGE
    # -------------------------------------------------------------
    print("\n--- 1. SÉLECTION DES FEATURES (SANS DATA LEAKAGE) ---")

    # Isolation temporaire du jeu de train pour la sélection ARIMA
    df_raw = pd.read_csv(config.DATA_RAW, parse_dates=True, index_col=0)
    split_train_len = len(df_raw) - config.TEST_SIZE
    train_only_path = config.BASE_DIR / "data" / "processed" / "temp_train_raw.csv"
    train_only_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw.iloc[:split_train_len].to_csv(train_only_path)

    # Sélection des lags sur le TRAIN uniquement
    best_lags = select_arima_lags(
        data_path=train_only_path,
        max_lags=config.MAX_LAGS
    )
    print(f"=> Lags optimaux retenus (sur Train set) : {best_lags}")

    # -------------------------------------------------------------
    # 2. PRÉPARATION DES DONNÉES
    # -------------------------------------------------------------
    print("\n--- 2. PRÉPARATION DES DONNÉES ---")

    data = prepare_supervised_data(
        data_path=config.DATA_RAW,
        lags=best_lags,
        test_size=config.TEST_SIZE,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    dates_test = data["dates_test"]
    y_train_diff = data["y_train_diff"]
    y_test_lag1 = data["y_test_lag1"]

    # -------------------------------------------------------------
    # 3. ENTRAÎNEMENT INITIAL DES MODÈLES
    # -------------------------------------------------------------
    print("\n--- 3. ENTRAÎNEMENT INITIAL ---")

    trained_models, target_scaler = train_and_save_models(
    X_train=X_train,
    y_train_diff=y_train_diff
    )
    print(f"=> Modèles entraînés : {list(trained_models.keys())}")
    # -------------------------------------------------------------
    # 4. SIMULATION WALK-FORWARD SÉQUENTIELLE
    # -------------------------------------------------------------
    print("\n--- 4. WALK-FORWARD SÉQUENTIEL ---")

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
        retrain_freq=60,
        X_train=X_train,
        y_train_diff=y_train_diff,
        lstm_names=["LSTM"],
        target_scaler=target_scaler
    )

    # Sauvegarde structurée des résultats
    output_dir = getattr(config, "RESULTS_DIR", config.BASE_DIR / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "results_walk_forward.csv"

    results_df.to_csv(out_file)
    print(f"\n=> Prédictions Walk-Forward sauvegardées dans : {out_file}")

    # -------------------------------------------------------------
    # 5. MÉTRIQUES ET SIMULATION FINANCIÈRE
    # -------------------------------------------------------------
    print("\n--- 5. MÉTRIQUES ET ÉVALUATION FINANCIÈRE ---")

    model_names = [
        "SVR",
        "MLP",
        "XGBoost",
        "LSTM",
        "Ensemble",
        "Benchmark (Marche Aléatoire)"
    ]

    predictions_dict = {
        name: results_df[name].values
        for name in model_names
        if name in results_df.columns
    }

    # Seuil sélectionné sur validation, appliqué figé sur le test final
    metrics_df, preds_final_df = compute_financial_metrics_validated(
        y_test=y_test,
        predictions=predictions_dict,
        dates_test=dates_test,
        y_test_lag1=y_test_lag1,
        val_ratio=0.25
    )
if __name__ == "__main__":
    main()