import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def hit_rate(y_true, y_pred):
    signs_true = np.sign(y_true)
    signs_pred = np.sign(y_pred)
    mask = (signs_pred != 0)
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(signs_true[mask] == signs_pred[mask])

def ic_binomial(k, n, alpha=0.05):
    if n == 0 or np.isnan(k):
        return (0.0, 1.0)
    p_lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    p_hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (p_lo * 100, p_hi * 100)

def diebold_mariano(y_true, pred1, pred2):
    d = np.abs(y_true - pred1) - np.abs(y_true - pred2)
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return np.nan, np.nan
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return dm_stat, p_value

def evaluate_macro_models(models, data, config):
    X_test = data["X_test"]
    y_test_brute = data["y_test_brute"]
    y_test_lag1 = data["y_test_lag1"]
    y_test_diff = data["y_test_diff"]
    y_train_diff = data["y_train_diff"]
    dates_test = data["dates_test"]
    feature_names = data["feature_names"]

    results = []
    predictions = {}
    n_test = len(y_test_diff)

    # Benchmarks
    rw_level_diff = np.zeros_like(y_test_diff)
    
    last_sign = np.sign(y_train_diff[-1]) if len(y_train_diff) > 0 else 1.0
    mean_abs_train = np.mean(np.abs(y_train_diff)) if len(y_train_diff) > 0 else 1.0
    rw_momentum_diff = np.full(n_test, last_sign * mean_abs_train)
    rng = np.random.default_rng(42)
    rw_random_diff = rng.choice([-1, 1], size=n_test) * np.mean(np.abs(y_test_diff))

    # Évaluation des Modèles
    for name, model in models.items():
        pred_diff = model.predict(X_test)
        pred_brute = y_test_lag1 + pred_diff

        mse = mean_squared_error(y_test_brute, pred_brute)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_brute, pred_brute)

        hr = hit_rate(y_test_diff, pred_diff)
        hr_k = int(round(hr * n_test / 100)) if not np.isnan(hr) else 0
        ic_lo, ic_hi = ic_binomial(hr_k, n_test)

        profits = np.where(pred_diff > 0, y_test_diff, np.where(pred_diff < 0, -y_test_diff, 0))
        profit_cum = np.sum(profits)
        sharpe = (np.mean(profits) / np.std(profits) * np.sqrt(12)) if np.std(profits) > 0 else 0

        dm_stat, dm_p = diebold_mariano(y_test_brute, pred_brute, y_test_lag1 + rw_level_diff)

        results.append({
            "Modèle": name,
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "Hit Rate (%)": round(hr, 2) if not np.isnan(hr) else "—",
            "IC 95% Hit Rate": f"[{ic_lo:.1f}% ; {ic_hi:.1f}%]",
            "Profit cumulé": round(profit_cum, 2),
            "Sharpe": round(sharpe, 2),
            "DM stat": round(dm_stat, 3) if not np.isnan(dm_stat) else "—",
            "DM p-value": round(dm_p, 4) if not np.isnan(dm_p) else "—"
        })
        predictions[name] = pred_brute

    # Évaluation Benchmarks
    for b_name, b_diff in [("RW (Niveau)", rw_level_diff), ("RW (Momentum)", rw_momentum_diff), ("RW (Aléatoire)", rw_random_diff)]:
        b_brute = y_test_lag1 + b_diff
        mse = mean_squared_error(y_test_brute, b_brute)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_brute, b_brute)
        hr = hit_rate(y_test_diff, b_diff)

        results.append({
            "Modèle": b_name,
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "Hit Rate (%)": round(hr, 2) if not np.isnan(hr) else "—",
            "IC 95% Hit Rate": "—",
            "Profit cumulé": 0.0,
            "Sharpe": 0.0,
            "DM stat": "—",
            "DM p-value": "—"
        })
        predictions[b_name] = b_brute

    df_results = pd.DataFrame(results)
    print("\n--- TABLEAU RÉCAPITULATIF DES PERFORMANCES ---")
    print(df_results.to_string(index=False))

    # Graphique de prévision
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test_brute, 'k--', linewidth=3, label="Taux Réel (MGA/USD)")
    colors = {"Ridge": "#1f77b4", "SVR": "#ff7f0e", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}
    for name, pred in predictions.items():
        if name in colors:
            plt.plot(dates_test, pred, color=colors[name], linewidth=2, label=f"Prédiction {name}")
    plt.title("Prévision Macroéconomique MGA/USD", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Taux de Change (MGA)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.MACRO_FORECAST_PLOT, dpi=300, bbox_inches='tight')
    plt.close()

    return df_results, predictions