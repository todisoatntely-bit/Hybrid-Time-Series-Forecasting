import pandas as pd
from sklearn.preprocessing import StandardScaler
import config_macro


def prepare_macro_data(data_path, test_size):
    # 1. Chargement des données
    df = pd.read_csv(data_path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # 2. Gestion IPC
    df['ipc'] = df['ipc'].interpolate(method='linear', limit_direction='forward')

    # 3. Features décalées (t-1)
    features_to_lag = ['mga_usd', 'brent', 'ipc', 'taux_directeur', 'reserves_mois_import']
    for col in features_to_lag:
        df[f'{col}_lag1'] = df[col].shift(1)

    df.dropna(inplace=True)

    # 4. Définition cible
    df['target_diff'] = df['mga_usd'] - df['mga_usd_lag1']
    feature_names = [f'{col}_lag1' for col in features_to_lag]

    X = df[feature_names].values
    y_brute = df['mga_usd'].values
    y_diff = df['target_diff'].values
    y_lag1 = df['mga_usd_lag1'].values
    dates = df.index

    # 5. Séparation Train/Test
    split_idx = len(df) - test_size

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train_diff = y_diff[:split_idx]
    y_test_brute = y_brute[split_idx:]
    y_test_lag1 = y_lag1[split_idx:]
    dates_test = dates[split_idx:]

    # 6. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    df_train = pd.DataFrame(X_train_scaled, columns=feature_names, index=dates[:split_idx])
    df_train['target_diff'] = y_train_diff
    df_train.to_csv(config_macro.DATA_PROCESSED_TRAIN)

    y_test_diff = y_test_brute - y_test_lag1

    return {
        "X_train": X_train_scaled,
        "y_train_diff": y_train_diff,
        "X_test": X_test_scaled,
        "y_test_brute": y_test_brute,
        "y_test_lag1": y_test_lag1,
        "y_test_diff": y_test_diff,
        "dates_test": dates_test,
        "feature_names": feature_names,
    }