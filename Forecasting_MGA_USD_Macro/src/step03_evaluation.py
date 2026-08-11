import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config_macro

def evaluate_macro_models(trained_models, X_test, y_test_brute, y_test_lag1, dates_test):
    results = []
    predictions = {}
    
    # 1. Benchmark de référence : Marche Aléatoire (Random Walk)
    benchmark_pred = y_test_lag1
    bm_mse = mean_squared_error(y_test_brute, benchmark_pred)
    bm_rmse = np.sqrt(bm_mse)
    bm_mae = mean_absolute_error(y_test_brute, benchmark_pred)
    actual_diff = y_test_brute - y_test_lag1

    results.append({
        "Modèle": "Benchmark (Marche Aléatoire)",
        "MSE": bm_mse,
        "RMSE": bm_rmse,
        "MAE": bm_mae,
        "Hit Rate (%)": 0.0
    })

    # 2. Évaluation des modèles d'IA
    for name, model in trained_models.items():
        pred_diff = model.predict(X_test)
        pred_brute = y_test_lag1 + pred_diff
        predictions[name] = pred_brute

        mse = mean_squared_error(y_test_brute, pred_brute)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_brute, pred_brute)

        # Précision directionnelle (Hit Rate)
        pred_direction = np.sign(pred_diff)
        actual_direction = np.sign(actual_diff)
        hit_rate = np.mean(pred_direction == actual_direction) * 100

        results.append({
            "Modèle": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Hit Rate (%)": hit_rate
        })

    # 3. Génération du tableau récapitulatif
    df_results = pd.DataFrame(results).set_index("Modèle").round(4)

    print("\n--- TABLEAU COMPARATIF DES PERFORMANCES MACRO (VOLET 2) ---")
    print(df_results.to_string())

    df_results.to_csv(config_macro.TABLES_DIR / "macro_results_metrics.csv")
    print(f"✅ Métriques sauvegardées dans : {config_macro.TABLES_DIR / 'macro_results_metrics.csv'}")

    # 4. Tracé et sauvegarde du graphique
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test_brute, label="Taux Réel (USD/MGA)", color="black", linewidth=2.5, linestyle="--")
    
    for name, pred in predictions.items():
        plt.plot(dates_test, pred, label=f"Prédiction {name}", alpha=0.8)

    plt.title("Prévision Macroéconomique USD/MGA sur Données Mensuelles", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Taux de Change (MGA)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_path = config_macro.FIGURES_DIR / "macro_forecast_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"✅ Graphique sauvegardé dans : {plot_path}")

    return df_results