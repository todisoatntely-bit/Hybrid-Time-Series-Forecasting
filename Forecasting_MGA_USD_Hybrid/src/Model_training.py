import os
import random
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from xgboost import XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import config


def set_reproducibility_seed(seed=42):
    """Fixe toutes les graines pour une reproductibilité maximale."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def chronological_train_val_split(X, y, val_ratio=0.15):
    """Split strictement chronologique (pas de mélange)."""
    split_idx = int(len(X) * (1 - val_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_and_save_models(X_train, y_train_diff):
    """
    Entraîne SVR, MLP, XGBoost et LSTM avec :
    - Scaling de la cible
    - Validation chronologique
    - Petite recherche d'hyperparamètres temporelle (SVR + XGBoost)
    """
    set_reproducibility_seed(42)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 0. SCALING DE LA CIBLE (critique pour SVR / MLP / LSTM)
    # ---------------------------------------------------------
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_diff.reshape(-1, 1)).ravel()

    # Sauvegarde du scaler de la cible
    joblib.dump(target_scaler, config.MODELS_DIR / "target_scaler.joblib")

    trained_models = {}

    # ---------------------------------------------------------
    # 1. SVR — petite recherche d'hyperparamètres temporelle
    # ---------------------------------------------------------
    print(" -> Optimisation + Entraînement du SVR...")

    X_tr, X_val, y_tr, y_val = chronological_train_val_split(X_train, y_train_scaled)

    param_dist_svr = {
        "C": [10, 50, 100, 200],
        "gamma": [0.01, 0.05, 0.1, 0.2],
        "epsilon": [0.01, 0.05, 0.1]
    }

    best_score = np.inf
    best_params_svr = None

    for params in ParameterSampler(param_dist_svr, n_iter=12, random_state=42):
        model = SVR(kernel="rbf", **params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        score = np.mean((pred - y_val) ** 2)
        if score < best_score:
            best_score = score
            best_params_svr = params

    print(f"    Meilleurs hyperparamètres SVR : {best_params_svr}")

    svr = SVR(kernel="rbf", **best_params_svr)
    svr.fit(X_train, y_train_scaled)
    joblib.dump(svr, config.MODELS_DIR / "svr_model.pkl")
    trained_models["SVR"] = svr

    # ---------------------------------------------------------
    # 2. MLP (sans early_stopping aléatoire)
    # ---------------------------------------------------------
    print(" -> Entraînement du MLP...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="tanh",
        solver="adam",
        max_iter=800,
        early_stopping=False,          # évite le split aléatoire de sklearn
        random_state=42
    )
    mlp.fit(X_train, y_train_scaled)
    joblib.dump(mlp, config.MODELS_DIR / "mlp_model.pkl")
    trained_models["MLP"] = mlp

    # ---------------------------------------------------------
    # 3. XGBoost — petite recherche d'hyperparamètres temporelle
    # ---------------------------------------------------------
    print(" -> Optimisation + Entraînement de XGBoost...")

    param_dist_xgb = {
        "n_estimators": [80, 120, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.8, 1.0]
    }

    best_score = np.inf
    best_params_xgb = None

    for params in ParameterSampler(param_dist_xgb, n_iter=12, random_state=42):
        model = XGBRegressor(objective="reg:squarederror", random_state=42, **params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        score = np.mean((pred - y_val) ** 2)
        if score < best_score:
            best_score = score
            best_params_xgb = params

    print(f"    Meilleurs hyperparamètres XGBoost : {best_params_xgb}")

    xgb = XGBRegressor(objective="reg:squarederror", random_state=42, **best_params_xgb)
    xgb.fit(X_train, y_train_scaled)
    joblib.dump(xgb, config.MODELS_DIR / "xgboost_model.pkl")
    trained_models["XGBoost"] = xgb

    # ---------------------------------------------------------
    # 4. LSTM — validation chronologique stricte
    # ---------------------------------------------------------
    print(" -> Entraînement du LSTM...")

    # Justification timesteps=1 :
    # Les lags des différences sont déjà construits explicitement comme features.
    # Le LSTM reçoit donc un vecteur de features stationnaires déjà lagguées.
    # Dans ce cadre, timesteps=1 est un choix délibéré (équivalent à un réseau
    # récurrent sur des features pré-ingénierées) plutôt qu'une fenêtre brute.

    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    split_idx = int(len(X_train_lstm) * 0.85)
    X_tr_lstm, X_val_lstm = X_train_lstm[:split_idx], X_train_lstm[split_idx:]
    y_tr_lstm, y_val_lstm = y_train_scaled[:split_idx], y_train_scaled[split_idx:]

    lstm = Sequential([
        LSTM(64, activation="tanh", input_shape=(1, X_train.shape[1])),
        Dropout(0.2),
        Dense(32, activation="tanh"),
        Dense(1)
    ])

    lstm.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True
    )

    lstm.fit(
        X_tr_lstm, y_tr_lstm,
        validation_data=(X_val_lstm, y_val_lstm),
        epochs=120,
        batch_size=32,
        callbacks=[early_stop],
        shuffle=False,          # critique pour les séries temporelles
        verbose=0
    )

    lstm.save(config.MODELS_DIR / "lstm_model.keras")
    trained_models["LSTM"] = lstm

    print(f"✅ Modèles entraînés et sauvegardés dans : {config.MODELS_DIR}")
    print(f"✅ Target scaler sauvegardé : {config.MODELS_DIR / 'target_scaler.joblib'}")

    # On retourne aussi le scaler pour le walk-forward
    return trained_models, target_scaler