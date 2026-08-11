import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config

def prepare_supervised_data(data_path, lags, test_size):
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    col_name = df.columns[0]
    
    # Création des features (lags)
    for lag in lags:
        df[f'lag_{lag}'] = df[col_name].shift(lag)
        
    df.dropna(inplace=True)
    
    # Variables explicatives (X) et Cible (y brute)
    X = df.drop(columns=[col_name]).values
    y = df[col_name].values
    dates = df.index
    
    # Split Train / Test
    split_idx = len(df) - test_size
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    # Normalisation des features X (Standardisation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # LA TRANSFORMATION "QUANT" : Création des variations (Delta y)
    # -------------------------------------------------------------
    # 1. Calcul des vrais taux de la veille
    y_train_lag1 = np.roll(y_train, 1)
    y_train_lag1[0] = y_train[0] 
    
    y_test_lag1 = np.roll(y_test, 1)
    y_test_lag1[0] = y_train[-1] # Le jour 1 du test dépend du dernier jour du train

    # 2. Les modèles ML devront prédire la variation (Diff), pas le prix
    y_train_diff = y_train - y_train_lag1
    
    # Sauvegarde des données processées (pour la reproductibilité)
    df_train = pd.DataFrame(X_train_scaled, columns=[f'lag_{l}' for l in lags], index=dates_train)
    df_train['target_diff'] = y_train_diff
    df_train.to_csv(config.DATA_PROCESSED_TRAIN)
    
    df_test = pd.DataFrame(X_test_scaled, columns=[f'lag_{l}' for l in lags], index=dates_test)
    df_test['target_brute'] = y_test
    df_test.to_csv(config.DATA_PROCESSED_TEST)

    return {
        "X_train": X_train_scaled, "y_train": y_train,
        "X_test": X_test_scaled, "y_test": y_test,
        "y_train_diff": y_train_diff,
        "y_test_lag1": y_test_lag1,
        "dates_test": dates_test
    }