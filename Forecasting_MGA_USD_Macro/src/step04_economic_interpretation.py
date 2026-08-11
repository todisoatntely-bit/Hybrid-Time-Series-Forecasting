import shap
import matplotlib.pyplot as plt
import config_macro

def interpret_economic_drivers(trained_models, X_train, feature_names):
    print(" -> Génération de l'interprétation économique (Valeurs de Shapley)...")
    
    # On choisit le Random Forest (ou XGBoost) car l'algorithme SHAP excelle 
    # pour décortiquer les modèles basés sur les arbres de décision.
    model = trained_models.get("Random Forest")
    
    if model is None:
        print("Erreur : Modèle Random Forest introuvable pour SHAP.")
        return

    # 1. L'Explainer : Il calcule la contribution marginale de chaque variable
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # 2. GRAPHIQUE 1 : L'Importance Globale (Le poids dans l'économie)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Importance Absolue des Variables Macroéconomiques", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot1_path = config_macro.FIGURES_DIR / "shap_importance_bar.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()

    # 3. GRAPHIQUE 2 : L'Impact Directionnel (Sens de causalité)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, show=False)
    plt.title("Directionnalité : Effet sur la dépréciation de l'Ariary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot2_path = config_macro.FIGURES_DIR / "shap_impact_beeswarm.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()

    print(f"✅ Graphiques SHAP d'interprétation sauvegardés dans : {config_macro.FIGURES_DIR}")