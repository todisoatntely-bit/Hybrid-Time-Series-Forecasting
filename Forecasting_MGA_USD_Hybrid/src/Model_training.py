import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Nouveaux imports pour le Deep Learning (LSTM)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import config

def train_and_save_models(X_train, y_train_diff):
    X_train_clean = X_train[1:]
    y_train_diff_clean = y_train_diff[1:]
    
    trained_models = {}

    print(" -> Entraînement du SVR...")
    svr = SVR(C=100, epsilon=0.01, gamma=0.01)
    svr.fit(X_train_clean, y_train_diff_clean)
    joblib.dump(svr, config.MODELS_DIR / "svr_model.pkl")
    trained_models["SVR"] = svr

    print(" -> Entraînement du MLP (Réseau de neurones)...")
    # Ajout du early_stopping pour éviter le warning que vous aviez eu !
    mlp = MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', 
                       max_iter=2000, early_stopping=True, random_state=42)
    mlp.fit(X_train_clean, y_train_diff_clean)
    joblib.dump(mlp, config.MODELS_DIR / "mlp_model.pkl")
    trained_models["MLP"] = mlp

    print(" -> Entraînement de XGBoost...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    xgb.fit(X_train_clean, y_train_diff_clean)
    joblib.dump(xgb, config.MODELS_DIR / "xgboost_model.pkl")
    trained_models["XGBoost"] = xgb

    # ==========================================
    # LE NOUVEAU MODÈLE : LSTM (Deep Learning)
    # ==========================================
    print(" -> Entraînement du LSTM (Deep Learning)...")
    
    # 1. Reshape 3D pour l'LSTM : [nombre_echantillons, pas_de_temps, variables]
    X_train_lstm = np.reshape(X_train_clean, (X_train_clean.shape[0], 1, X_train_clean.shape[1]))
    
    # 2. Construction de l'architecture du réseau
    lstm = Sequential([
        LSTM(50, activation='relu', input_shape=(1, X_train_clean.shape[1])),
        Dense(1) # Couche de sortie : 1 seule prédiction (la variation)
    ])
    lstm.compile(optimizer='adam', loss='mse')
    
    # 3. Entraînement avec "patience" (s'arrête s'il n'apprend plus)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm.fit(X_train_lstm, y_train_diff_clean, epochs=100, batch_size=16, 
             validation_split=0.1, callbacks=[early_stop], verbose=0)
    
    # 4. Sauvegarde (Keras utilise le format .keras au lieu de .pkl)
    lstm.save(config.MODELS_DIR / "lstm_model.keras")
    trained_models["LSTM"] = lstm

    print(f"✅ Tous les modèles sont sauvegardés dans : {config.MODELS_DIR}")
    
    return trained_models