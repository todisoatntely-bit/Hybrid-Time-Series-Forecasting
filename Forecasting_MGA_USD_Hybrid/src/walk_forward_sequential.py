import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def walk_forward_predict(
    trained_models,
    X_test,
    y_test_lag1,
    y_test=None,
    retrain_models=None,
    retrain_freq=None,
    X_train=None,
    y_train_diff=None,
    lstm_names=None,
    target_scaler=None
):
    """
    Walk-forward séquentiel 1-step-ahead avec support LSTM
    et inverse-transform de la cible scalée.
    """
    if lstm_names is None:
        lstm_names = ["LSTM"]

    n_test = len(X_test)
    predictions = {name: [] for name in trained_models.keys()}
    predictions["Benchmark (Marche Aléatoire)"] = []

    # Conversions sécurisées
    X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
    index_test = X_test.index if isinstance(X_test, pd.DataFrame) else pd.RangeIndex(n_test)
    y_lag1_values = y_test_lag1.values if isinstance(y_test_lag1, pd.Series) else np.array(y_test_lag1)

    y_test_values = None
    if y_test is not None:
        y_test_values = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)

    print(f"=== Walk-Forward Séquentiel : {n_test} pas à prédire ===")

    for i in range(n_test):
        if i % 200 == 0 or i == n_test - 1:
            print(f"  Progression : {i+1}/{n_test}")

        x_curr = X_test_values[i:i+1]
        lag1_curr = y_lag1_values[i]

        for name, model in trained_models.items():

            # --- 1. Réentraînement périodique (optionnel) ---
            if (
                retrain_models
                and retrain_models.get(name, False)
                and retrain_freq
                and i > 0
                and i % retrain_freq == 0
                and X_train is not None
                and y_train_diff is not None
                and y_test_values is not None
            ):
                try:
                    x_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.array(X_train)
                    y_train_diff_arr = (
                        y_train_diff.values if isinstance(y_train_diff, pd.Series)
                        else np.array(y_train_diff)
                    )

                    # Différences observées sur le test déjà vu
                    y_test_diff_seen = y_test_values[:i] - y_lag1_values[:i]
                    x_seen = np.vstack([x_train_arr, X_test_values[:i]])
                    y_seen = np.concatenate([y_train_diff_arr, y_test_diff_seen])

                    # Si un target_scaler existe, on scale y_seen
                    if target_scaler is not None:
                        y_seen = target_scaler.transform(y_seen.reshape(-1, 1)).ravel()

                    if name in lstm_names:
                        x_seen_lstm = np.reshape(x_seen, (x_seen.shape[0], 1, x_seen.shape[1]))
                        model.fit(x_seen_lstm, y_seen, epochs=5, batch_size=32, verbose=0)
                    else:
                        model.fit(x_seen, y_seen)

                    print(f"    [RETRAIN] Modèle {name} réentraîné à l'étape {i}")
                except Exception as e:
                    print(f"    [WARN] Échec du réentraînement pour {name} à l'étape {i}: {e}")

            # --- 2. Prédiction ---
            if name in lstm_names:
                x_lstm = np.reshape(x_curr, (1, 1, x_curr.shape[1]))
                pred_scaled = float(model(x_lstm, training=False).numpy()[0, 0])
            else:
                pred_scaled = float(model.predict(x_curr)[0])

            # Inverse-transform de la différence prédite
            if target_scaler is not None:
                pred_diff = target_scaler.inverse_transform(
                    np.array([[pred_scaled]])
                )[0, 0]
            else:
                pred_diff = pred_scaled

            # Reconstruction du niveau
            pred_level = lag1_curr + pred_diff
            predictions[name].append(pred_level)

        # Benchmark marche aléatoire
        predictions["Benchmark (Marche Aléatoire)"].append(lag1_curr)
    # --- Ensemble (moyenne simple des 4 modèles) ---
    model_cols = [name for name in trained_models.keys()]
    if len(model_cols) >= 2:
        ensemble_preds = np.mean([predictions[name] for name in model_cols], axis=0)
        predictions["Ensemble"] = ensemble_preds.tolist()
    results = pd.DataFrame(predictions, index=index_test)

    # --- 3. Évaluation rapide (si y_test fourni) ---
    if y_test_values is not None:
        print("\n=== Synthèse des performances Walk-Forward ===")
        print(f"{'Modèle':<28} | {'RMSE':>8} | {'MAE':>8} | {'Hit Rate':>8}")
        print("-" * 62)
        for col in results.columns:
            rmse = np.sqrt(mean_squared_error(y_test_values, results[col]))
            mae = mean_absolute_error(y_test_values, results[col])

            actual_dir = np.sign(y_test_values - y_lag1_values)
            pred_dir = np.sign(results[col].values - y_lag1_values)
            valid = actual_dir != 0
            hit_rate = (actual_dir[valid] == pred_dir[valid]).mean() if valid.sum() > 0 else 0.0

            print(f"{col:<28} | {rmse:>8.4f} | {mae:>8.4f} | {hit_rate:>7.2%}")

    return results