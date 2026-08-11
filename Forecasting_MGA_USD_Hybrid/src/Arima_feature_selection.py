import pandas as pd
import pmdarima as pm
import json
import config

def select_arima_lags(data_path, max_lags=10):
    """
    Détermine la structure ARIMA via le critère BIC.
    Retourne la liste des retards (lags) optimaux.
    """
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    ts = df.iloc[:, 0]

    print("\nRecherche de la structure ARIMA optimale (Objectif : BIC)...")
    # Utilisation d'auto_arima pour trouver l'ordre (p, d, q)
    model = pm.auto_arima(
        ts,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        d=1, # On force d=1 (car on sait que la série est non stationnaire)
        seasonal=False,
        information_criterion='bic',
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False # Mis sur False pour un affichage plus propre dans le pipeline
    )
    
    best_p, best_d, best_q = model.order
    print(f"Meilleur modèle ARIMA trouvé : ARIMA({best_p}, {best_d}, {best_q})")
    
    # Si p=0, on force l'utilisation du lag 1 au minimum pour le Machine Learning
    lags_list = list(range(1, best_p + 1)) if best_p > 0 else [1]
    
    # Sauvegarde des paramètres (optionnel mais très pro)
    features_config = {
        "arima_order": {"p": best_p, "d": best_d, "q": best_q},
        "selected_lags": lags_list
    }
    
    json_path = config.BASE_DIR / "data" / "processed" / "selected_lags.json"
    with open(json_path, 'w') as f:
        json.dump(features_config, f, indent=4)
        
    return lags_list