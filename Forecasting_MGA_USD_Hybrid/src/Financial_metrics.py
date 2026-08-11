import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config

def calculate_directional_accuracy(y_true, y_pred, y_lag1):
    """Calcule le Hit Rate (précision directionnelle)."""
    actual_direction = np.sign(y_true - y_lag1)
    predicted_direction = np.sign(y_pred - y_lag1)
    correct_predictions = (actual_direction == predicted_direction).sum()
    # On exclut les cas où il n'y a aucune variation (diff = 0)
    total_valid = (actual_direction != 0).sum()
    return (correct_predictions / total_valid) * 100 if total_valid > 0 else 0

def simulate_trading(y_true, y_pred, y_lag1):
    """Simule une stratégie de trading basique."""
    predicted_direction = np.sign(y_pred - y_lag1)
    actual_returns = (y_true - y_lag1) / y_lag1
    strategy_returns = predicted_direction * actual_returns
    
    cum_profit = strategy_returns.sum() * 100
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
    
    return cum_profit, sharpe_ratio

def compute_financial_metrics(y_test, predictions, dates_test):
    results = []
    y_test_lag1 = predictions["Benchmark (Marche Aléatoire)"]
    
    # Création du DataFrame pour les CSV
    df_preds = pd.DataFrame(index=dates_test)
    df_preds['Vrai_Prix'] = y_test
    
    # 1. Calcul des métriques
    for name, y_pred in predictions.items():
        df_preds[f'Pred_{name}'] = y_pred
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        hit_rate = calculate_directional_accuracy(y_test, y_pred, y_test_lag1)
        cum_profit, sharpe = simulate_trading(y_test, y_pred, y_test_lag1)
        
        results.append({
            "Modèle": name,
            "MSE": mse, "RMSE": rmse, "MAE": mae,
            "Hit Rate (%)": hit_rate,
            "Profit Cumulé (%)": cum_profit,
            "Ratio de Sharpe": sharpe
        })

    # Sauvegarde des tableaux CSV
    df_results = pd.DataFrame(results).set_index("Modèle")
    df_results.to_csv(config.TABLES_DIR / "results_metrics.csv")
    df_preds.to_csv(config.TABLES_DIR / "predictions.csv")
    
    # 2. Affichage Console
    print("\n--- TABLEAU RÉCAPITULATIF DES PERFORMANCES ---")
    print(df_results.round(4).to_string())

    # 3. Génération du graphique des prédictions
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, y_test, label="Vrai Taux (MGA/USD)", color="black", linewidth=2.5)
    
    colors = {"SVR": "blue", "MLP": "green", "XGBoost": "red", "Benchmark (Marche Aléatoire)": "gray"}
    for name, y_pred in predictions.items():
        plt.plot(dates_test, y_pred, label=f"Prédiction {name}", 
                 color=colors.get(name, "orange"), linestyle="--" if "Benchmark" in name else "-")
                 
    plt.title("Prévisions Hybrides du Taux de Change USD/MGA (Validation Out-of-Sample)")
    plt.xlabel("Date")
    plt.ylabel("Taux de change (Ariary)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig_path = config.FIGURES_DIR / "forecast_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Graphique sauvegardé dans : {fig_path}")