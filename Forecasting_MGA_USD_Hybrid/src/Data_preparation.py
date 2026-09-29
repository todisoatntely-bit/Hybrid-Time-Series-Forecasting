import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config


def prepare_supervised_data(data_path, lags, test_size):
    """
    Prépare les données pour un modèle ML stationnaire (prédiction des variations Delta y).
    """
    df = pd.read_csv(data_path, parse_dates=[0], index_col=0)
    col_name = df.columns[0]

    # 1. Calcul de la variation journalière (Cible stationnaire)
    df['target_diff'] = df[col_name].diff()

    # 2. Création des features stationnaires (lags des variations passées)
    for lag in lags:
        df[f'lag_diff_{lag}'] = df['target_diff'].shift(lag)

    # Alignement du prix de la veille pour la reconstruction finale : y_t = y_{t-1} + pred_diff
    df['y_lag1'] = df[col_name].shift(1)

    # Nettoyage des NaN générés par les shifts
    df.dropna(inplace=True)

    # Extraction des matrices
    feature_cols = [f'lag_diff_{lag}' for lag in lags]
    X = df[feature_cols].values
    y_diff = df['target_diff'].values
    y_brute = df[col_name].values
    y_lag1 = df['y_lag1'].values
    dates = df.index

    # Split Train / Test
    split_idx = len(df) - test_size
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_diff, y_test_diff = y_diff[:split_idx], y_diff[split_idx:]
    y_train, y_test = y_brute[:split_idx], y_brute[split_idx:]
    y_train_lag1, y_test_lag1 = y_lag1[:split_idx], y_lag1[split_idx:]

    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    # Normalisation des features X
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Persistance du scaler
    scaler_path = config.BASE_DIR / "data" / "processed" / "scaler.joblib"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    # Sauvegarde des DataFrames structurés
    df_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=dates_train)
    df_train['target_diff'] = y_train_diff
    df_train['target_brute'] = y_train
    df_train.to_csv(config.DATA_PROCESSED_TRAIN)

    df_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=dates_test)
    df_test['target_diff'] = y_test_diff
    df_test['target_brute'] = y_test
    df_test['y_lag1'] = y_test_lag1
    df_test.to_csv(config.DATA_PROCESSED_TEST)

    return {
        "X_train": X_train_scaled, "y_train": y_train, "y_train_diff": y_train_diff,
        "X_test": X_test_scaled, "y_test": y_test, "y_test_diff": y_test_diff,
        "y_test_lag1": y_test_lag1, "dates_test": dates_test, "scaler": scaler
    }