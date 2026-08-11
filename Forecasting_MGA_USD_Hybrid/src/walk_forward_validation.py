import pandas as pd
import numpy as np

def evaluate_predictions(trained_models, X_test, y_test_lag1):
    predictions = {}
    
    for name, model in trained_models.items():
        if name == "LSTM":
            # 1. Passage en 3D obligatoire pour l'LSTM
            X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            # 2. Prédiction et aplatissement (pour repasser en tableau classique)
            pred_diff = model.predict(X_test_lstm, verbose=0).flatten()
        else:
            # SVR, MLP, et XGBoost prennent des données normales 2D
            pred_diff = model.predict(X_test)
        
        # RECONSTRUCTION : Prix prédit = Prix de la veille + Variation prédite
        predictions[name] = y_test_lag1 + pred_diff

    # Ajout du Benchmark : Marche Aléatoire
    predictions["Benchmark (Marche Aléatoire)"] = y_test_lag1

    return predictions