"""
Walk-Forward Séquentiel pour prédiction de séries temporelles financières.
Prédit un pas de temps à la fois avec reconstruction des niveaux.
Compatible avec les modèles du pipeline Hybrid-Time-Series-Forecasting.
"""

import pandas as pd
import numpy as np
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
    lstm_names=None
):
    """
    Walk-forward séquentiel : prédit un pas de temps à la fois.

    Parameters
    ----------
    trained_models : dict
        {nom: modèle} déjà entraîné (SVR, XGBoost, LSTM, etc.)
    X_test : pd.DataFrame ou np.ndarray
        Features de test (ORDRE CHRONOLOGIQUE obligatoire)
    y_test_lag1 : pd.Series ou np.ndarray
        Valeur de la veille pour reconstruction du niveau
    y_test : pd.Series ou np.ndarray, optional
        Valeurs réelles (niveau) pour calcul des métriques
    retrain_models : dict, optional
        {nom: bool} indiquant si le modèle doit être réentraîné périodiquement
    retrain_freq : int, optional
        Fréquence de réentraînement (nombre d'observations)
    X_train : pd.DataFrame, optional
        Données d'entraînement initiales (pour réentraînement)
    y_train_diff : pd.Series, optional
        Target d'entraînement initiale (différences)
    lstm_names : list, optional
        Liste des noms de modèles LSTM (ex: ["LSTM"])

    Returns
    -------
    pd.DataFrame
        Prédictions de chaque modèle + benchmark naïf
    """
    if lstm_names is None:
        lstm_names = ["LSTM"]

    n_test = len(X_test)
    predictions = {name: [] for name in trained_models.keys()}
    predictions["Benchmark (Marche Aléatoire)"] = []

    # Conversion numpy pour indexation rapide
    if isinstance(X_test, pd.DataFrame):
        X_test_values = X_test.values
        index_test = X_test.index
    else:
        X_test_values = np.array(X_test)
        index_test = pd.RangeIndex(n_test)

    if isinstance(y_test_lag1, pd.Series):
        y_lag1_values = y_test_lag1.values
    else:
        y_lag1_values = np.array(y_test_lag1)

    if y_test is not None:
        if isinstance(y_test, pd.Series):
            y_test_values = y_test.values
        else:
            y_test_values = np.array(y_test)
    else:
        y_test_values = None

    print(f"=== Walk-Forward Séquentiel : {n_test} pas ===")

    for i in range(n_test):
        if i % 50 == 0 or i == n_test - 1:
            print(f"  Progression : {i+1}/{n_test}")

        x_curr = X_test_values[i:i+1]
        lag1_curr = y_lag1_values[i]

        for name, model in trained_models.items():
            # --- Réentraînement périodique (optionnel, modèles rapides uniquement) ---
            if (
                retrain_models
                and retrain_models.get(name, False)
                and retrain_freq
                and i > 0
                and i % retrain_freq == 0
                and X_train is not None
                and y_train_diff is not None
            ):
                try:
                    if isinstance(X_test, pd.DataFrame):
                        x_seen = pd.concat([X_train, X_test.iloc[:i]])
                        # Pour y, on reconstruit les différences vues
                        y_seen = pd.concat([y_train_diff, pd.Series(y_test_values[:i] - y_lag1_values[:i])])
                    else:
                        x_seen = np.vstack([X_train, X_test_values[:i]])
                        y_seen = np.concatenate([
                            np.array(y_train_diff),
                            y_test_values[:i] - y_lag1_values[:i]
                        ])
                    model.fit(x_seen, y_seen)
                    print(f"    [RETRAIN] {name} réentraîné à l'étape {i}")
                except Exception as e:
                    print(f"    [WARN] Réentraînement échoué pour {name}: {e}")

            # --- Prédiction ---
            if name in lstm_names:
                # LSTM : reshape (samples, timesteps, features)
                # Ici timesteps=1 car les lags sont déjà dans les features
                x_lstm = np.reshape(x_curr, (1, 1, x_curr.shape[1]))
                pred_diff = model.predict(x_lstm, verbose=0).flatten()[0]
            else:
                pred_diff = model.predict(x_curr)[0]

            # Reconstruction du niveau : Prix = Prix_veille + Variation_prédite
            pred_level = lag1_curr + pred_diff
            predictions[name].append(pred_level)

        # Benchmark naïf : demain = aujourd'hui
        predictions["Benchmark (Marche Aléatoire)"].append(lag1_curr)

    # Assemblage
    results = pd.DataFrame(predictions, index=index_test)

    # Métriques de base
    if y_test_values is not None:
        print("\n=== Métriques Walk-Forward ===")
        print(f"{'Modèle':<25} | {'RMSE':>8} | {'MAE':>8} | {'Hit Rate':>8}")
        print("-" * 60)
        for col in results.columns:
            rmse = np.sqrt(mean_squared_error(y_test_values, results[col]))
            mae = mean_absolute_error(y_test_values, results[col])
            # Hit Rate : direction prédite vs direction réelle
            actual_dir = np.sign(np.diff(y_test_values, prepend=y_test_values[0]))
            pred_dir = np.sign(np.diff(results[col].values, prepend=results[col].iloc[0]))
            hit_rate = np.mean(actual_dir == pred_dir)
            print(f"{col:<25} | {rmse:>8.4f} | {mae:>8.4f} | {hit_rate:>7.2%}")

    return results
