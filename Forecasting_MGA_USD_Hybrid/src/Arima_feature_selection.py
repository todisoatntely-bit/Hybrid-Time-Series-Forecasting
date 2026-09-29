import json
import pandas as pd
import pmdarima as pm
import config


def select_arima_lags(data_path, max_lags=10):
    """
    Détermine la structure ARIMA via le critère BIC sur le jeu de données fourni.
    Retourne la liste des retards (lags) optimaux.
    """
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    ts = df.iloc[:, 0]

    print("\nRecherche de la structure ARIMA optimale (Objectif : BIC)...")

    # Utilisation du paramètre max_lags pour limiter max_p et max_q
    model = pm.auto_arima(
        ts,
        start_p=0, start_q=0,
        max_p=max_lags, max_q=max_lags,  # Utilisation dynamique de max_lags
        d=1,  # Série non stationnaire
        seasonal=False,
        information_criterion='bic',
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )

    best_p, best_d, best_q = model.order
    print(f"Meilleur modèle ARIMA trouvé : ARIMA({best_p}, {best_d}, {best_q})")

    # Si p=0, on se rabat au minimum sur le lag 1 (ou p+q selon besoin)
    max_lag_selected = max(best_p, 1)
    lags_list = list(range(1, max_lag_selected + 1))

    features_config = {
        "arima_order": {"p": best_p, "d": best_d, "q": best_q},
        "selected_lags": lags_list
    }

    # Sécurité pour créer le dossier 'processed' s'il n'existe pas
    json_path = config.BASE_DIR / "data" / "processed" / "selected_lags.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(features_config, f, indent=4)

    return lags_list