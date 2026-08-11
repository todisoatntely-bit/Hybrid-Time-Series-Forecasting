import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import config_macro

def train_macro_models(X_train, y_train_diff):
    trained_models = {}

    # 1. Le Modèle Économétrique Classique : Ridge Regression (Régression Linéaire Pénalisée)
    print(" -> Entraînement de la Régression Ridge (Baseline Linéaire)...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train_diff)
    joblib.dump(ridge, config_macro.MODELS_DIR / "macro_ridge_model.pkl")
    trained_models["Ridge"] = ridge

    # 2. Le Modèle Robuste au Bruit : SVR
    print(" -> Entraînement du SVR...")
    svr = SVR(C=10, epsilon=0.1, kernel='rbf')
    svr.fit(X_train, y_train_diff)
    joblib.dump(svr, config_macro.MODELS_DIR / "macro_svr_model.pkl")
    trained_models["SVR"] = svr

    # 3. Le Modèle d'Interactions Non-Linéaires : Random Forest
    print(" -> Entraînement du Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train_diff)
    joblib.dump(rf, config_macro.MODELS_DIR / "macro_rf_model.pkl")
    trained_models["Random Forest"] = rf

    # 4. Le Champion des Compétitions : XGBoost
    print(" -> Entraînement de XGBoost...")
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    xgb.fit(X_train, y_train_diff)
    joblib.dump(xgb, config_macro.MODELS_DIR / "macro_xgboost_model.pkl")
    trained_models["XGBoost"] = xgb

    print(f"✅ Modèles macroéconomiques sauvegardés dans : {config_macro.MODELS_DIR}")
    
    return trained_models