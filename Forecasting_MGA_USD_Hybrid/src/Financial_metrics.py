import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config


# ============================================================
# 1. HIT RATE — PRÉCISION DIRECTIONNELLE
# ============================================================

def calculate_directional_accuracy(y_true, y_pred, y_lag1):
    """
    Calcule le Hit Rate (précision directionnelle).
    Exclut les jours où la variation réelle est nulle.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_lag1 = np.asarray(y_lag1)

    actual_direction = np.sign(y_true - y_lag1)
    predicted_direction = np.sign(y_pred - y_lag1)

    valid = actual_direction != 0

    if valid.sum() == 0:
        return 0.0

    correct_predictions = (actual_direction[valid] == predicted_direction[valid]).sum()
    return (correct_predictions / valid.sum()) * 100


# ============================================================
# 2. SIMULATION DE TRADING
# ============================================================

def simulate_trading(y_true, y_pred, y_lag1, transaction_cost=0.001,
                     risk_free_rate=0.08, confidence_threshold=0.0):
    """
    Stratégie directionnelle avec filtre de confiance.
    Si |pred_diff| < threshold → position = 0 (on ne trade pas).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_lag1 = np.asarray(y_lag1)

    pred_diff = y_pred - y_lag1
    predicted_direction = np.sign(pred_diff)

    # Filtre de confiance
    predicted_direction[np.abs(pred_diff) < confidence_threshold] = 0.0

    actual_returns = (y_true - y_lag1) / y_lag1
    gross_returns = predicted_direction * actual_returns

    # Frais uniquement quand on change de position
    position_changes = np.diff(predicted_direction, prepend=0) != 0
    costs = position_changes * transaction_cost
    net_returns = gross_returns - costs

    # Profit cumulé composé
    cum_profit = (np.prod(1 + net_returns) - 1) * 100

    # Sharpe
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = net_returns - rf_daily
    std_returns = net_returns.std(ddof=1)

    if std_returns > 0:
        sharpe_ratio = (excess_returns.mean() / std_returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    return cum_profit, sharpe_ratio


# ============================================================
# 3. MÉTRIQUES FINANCIÈRES (VERSION ORIGINALE — conservée)
# ============================================================
# Conservée pour référence / comparaison avant-après, mais NE DOIT PLUS être
# utilisée pour les résultats définitifs du mémoire : le "seuil optimal" y
# est en réalité sélectionné ex post sur le test lui-même (data snooping).
# Utiliser compute_financial_metrics_validated() ci-dessous à la place.

def compute_financial_metrics(y_test, predictions, dates_test, y_test_lag1):
    """
    Calcule et enregistre les métriques d'évaluation + test de plusieurs seuils.

    ATTENTION : les 4 seuils sont tous évalués directement sur y_test. Choisir
    a posteriori le seuil au meilleur Sharpe parmi ces résultats revient à
    sélectionner ex post sur le test — voir compute_financial_metrics_validated().
    """
    y_test = np.asarray(y_test)
    y_test_lag1 = np.asarray(y_test_lag1)

    if len(y_test) != len(y_test_lag1) or len(y_test) != len(dates_test):
        raise ValueError("Les dimensions de y_test, y_test_lag1 et dates_test doivent concorder.")

    results = []
    df_preds = pd.DataFrame(index=dates_test)
    df_preds["Vrai_Prix"] = y_test

    thresholds = [0.0, 2.0, 5.0, 10.0]  # 0 = pas de filtre

    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred)

        if len(y_pred) != len(y_test):
            raise ValueError(f"Longueur incorrecte pour {name} : {len(y_pred)} vs {len(y_test)}.")

        df_preds[f"Pred_{name}"] = y_pred

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        hit_rate = calculate_directional_accuracy(y_test, y_pred, y_test_lag1)

        # Test de plusieurs seuils de confiance
        for thresh in thresholds:
            cum_profit, sharpe = simulate_trading(
                y_test,
                y_pred,
                y_test_lag1,
                confidence_threshold=thresh
            )

            results.append({
                "Model": f"{name} (seuil={thresh})",
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "Hit Rate (%)": hit_rate,
                "Profit Net (%)": cum_profit,
                "Sharpe Ratio": sharpe
            })

    results_df = pd.DataFrame(results)

    print("\n=== MÉTRIQUES FINANCIÈRES (NETTES DE FRAIS) ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Sauvegarde
    output_dir = getattr(config, "RESULTS_DIR", config.BASE_DIR / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "results_metrics.csv"
    preds_path = output_dir / "predictions.csv"
    plot_path = output_dir / "forecast_comparison.png"

    results_df.to_csv(metrics_path, index=False)
    df_preds.to_csv(preds_path, index=True)

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test, label="Prix réel", linewidth=2.5, color="black")

    line_styles = {
        "SVR": "-",
        "MLP": "--",
        "XGBoost": "-.",
        "LSTM": ":",
        "Ensemble": "-",
        "Benchmark (Marche Aléatoire)": "--"
    }

    for name in predictions:
        plt.plot(
            dates_test,
            predictions[name],
            label=name,
            linestyle=line_styles.get(name, "-"),
            linewidth=1.5
        )

    plt.title("Comparaison des prévisions USD/MGA", fontsize=14)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Taux de change USD/MGA", fontsize=11)
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n=> Métriques sauvegardées dans : {metrics_path}")
    print(f"=> Prédictions sauvegardées dans : {preds_path}")
    print(f"=> Graphique sauvegardé dans : {plot_path}")

    return results_df


# ============================================================
# 4. SÉLECTION DU SEUIL SUR VALIDATION (NOUVEAU)
# ============================================================

def select_threshold_on_validation(y_val, y_val_pred, y_val_lag1, thresholds,
                                    selection_metric="sharpe"):
    """
    Choisit, pour UN modèle, le seuil qui maximise le Sharpe (ou le profit)
    sur le bloc de validation — jamais sur le test final.

    Retourne (best_threshold, best_score, scores_par_seuil).
    """
    y_val = np.asarray(y_val)
    y_val_pred = np.asarray(y_val_pred)
    y_val_lag1 = np.asarray(y_val_lag1)

    best_threshold = thresholds[0]
    best_score = -np.inf
    scores = {}

    for thresh in thresholds:
        profit, sharpe = simulate_trading(y_val, y_val_pred, y_val_lag1, confidence_threshold=thresh)
        score = sharpe if selection_metric == "sharpe" else profit
        scores[thresh] = {"profit": profit, "sharpe": sharpe}

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score, scores


# ============================================================
# 5. MÉTRIQUES FINANCIÈRES — VERSION CORRIGÉE (À UTILISER)
# ============================================================

def compute_financial_metrics_validated(y_test, predictions, dates_test, y_test_lag1,
                                          val_ratio=0.25,
                                          thresholds=(0.0, 2.0, 5.0, 10.0),
                                          selection_metric="sharpe"):
    """
    Version corrigée de compute_financial_metrics() : le seuil de confiance
    est choisi sur un bloc de VALIDATION chronologique distinct du bloc de
    TEST final, pour éviter la sélection ex post sur le test (data snooping).

    Découpage chronologique :
        [ ---------- test complet (n_test obs) ---------- ]
        [ -- validation (val_ratio) -- ][ -- test final -- ]

    Toutes les métriques rapportées (RMSE, MAE, Hit Rate, Profit, Sharpe)
    portent UNIQUEMENT sur le bloc "test final", avec le seuil figé choisi
    sur la validation.
    """
    y_test = np.asarray(y_test)
    y_test_lag1 = np.asarray(y_test_lag1)
    dates_test = pd.DatetimeIndex(dates_test)

    n_total = len(y_test)
    if len(y_test_lag1) != n_total or len(dates_test) != n_total:
        raise ValueError("Les dimensions de y_test, y_test_lag1 et dates_test doivent concorder.")

    n_val = int(np.floor(n_total * val_ratio))
    if n_val < 10:
        raise ValueError(f"Bloc de validation trop petit (n_val={n_val}). "
                          f"Réduisez val_ratio ou vérifiez la taille du test.")
    n_final = n_total - n_val

    print("=== Découpage validation / test final ===")
    print(f"  Total test         : {n_total} obs ({dates_test[0].date()} -> {dates_test[-1].date()})")
    print(f"  Validation (seuil) : {n_val} obs ({dates_test[0].date()} -> {dates_test[n_val-1].date()})")
    print(f"  Test final (résult): {n_final} obs ({dates_test[n_val].date()} -> {dates_test[-1].date()})")

    # Découpage des vérités terrain
    y_val, y_final = y_test[:n_val], y_test[n_val:]
    y_val_lag1, y_final_lag1 = y_test_lag1[:n_val], y_test_lag1[n_val:]
    dates_final = dates_test[n_val:]

    results = []
    df_preds_final = pd.DataFrame(index=dates_final)
    df_preds_final["Vrai_Prix"] = y_final

    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred)
        if len(y_pred) != n_total:
            raise ValueError(f"Longueur incorrecte pour {name} : {len(y_pred)} vs {n_total}.")

        y_pred_val, y_pred_final = y_pred[:n_val], y_pred[n_val:]

        # --- 1. Sélection du seuil UNIQUEMENT sur la validation ---
        best_thresh, best_val_score, _ = select_threshold_on_validation(
            y_val, y_pred_val, y_val_lag1, list(thresholds), selection_metric
        )
        print(f"  -> {name}: seuil retenu = {best_thresh} "
              f"({selection_metric}_validation = {best_val_score:.4f})")

        df_preds_final[f"Pred_{name}"] = y_pred_final

        # --- 2. Évaluation figée sur le test final ---
        mse = mean_squared_error(y_final, y_pred_final)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_final, y_pred_final)
        hit_rate = calculate_directional_accuracy(y_final, y_pred_final, y_final_lag1)
        profit, sharpe = simulate_trading(
            y_final, y_pred_final, y_final_lag1, confidence_threshold=best_thresh
        )

        results.append({
            "Model": name,
            "Seuil retenu (validation)": best_thresh,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Hit Rate (%)": hit_rate,
            "Profit Net (%)": profit,
            "Sharpe Ratio": sharpe,
            "n_validation": n_val,
            "n_test_final": n_final,
        })

    results_df = pd.DataFrame(results)

    print("\n=== MÉTRIQUES FINANCIÈRES — TEST FINAL UNIQUEMENT (seuil fixé sur validation) ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Sauvegarde ---
    output_dir = getattr(config, "RESULTS_DIR", config.BASE_DIR / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "results_metrics_validated.csv", index=False)
    df_preds_final.to_csv(output_dir / "predictions_test_final.csv", index=True)

    # Plot du test final uniquement (avec seuil déjà figé)
    plt.figure(figsize=(14, 7))
    plt.plot(dates_final, y_final, label="Prix réel", linewidth=2.5, color="black")

    line_styles = {
        "SVR": "-",
        "MLP": "--",
        "XGBoost": "-.",
        "LSTM": ":",
        "Ensemble": "-",
    }
    for name in predictions:
        plt.plot(
            dates_final,
            df_preds_final[f"Pred_{name}"],
            label=name,
            linestyle=line_styles.get(name, "-"),
            linewidth=1.5
        )

    plt.title("Prévisions USD/MGA — Test final (seuil sélectionné hors test)", fontsize=14)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Taux de change USD/MGA", fontsize=11)
    plt.legend(loc="best")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "forecast_comparison_validated.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n=> Métriques (validées)     : {output_dir / 'results_metrics_validated.csv'}")
    print(f"=> Prédictions (test final)  : {output_dir / 'predictions_test_final.csv'}")
    print(f"=> Graphique (test final)    : {output_dir / 'forecast_comparison_validated.png'}")

    return results_df, df_preds_final