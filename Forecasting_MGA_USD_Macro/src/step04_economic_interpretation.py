import shap
import matplotlib.pyplot as plt
import config_macro


def interpret_economic_drivers(trained_models, X_test, feature_names):
    print(" -> Génération de l'interprétation économique (SHAP) sur données TEST...")

    # On fait le SHAP sur les deux modèles à base d'arbres
    tree_models = {
        "Random Forest": trained_models.get("Random Forest"),
        "XGBoost": trained_models.get("XGBoost")
    }

    for model_name, model in tree_models.items():
        if model is None:
            print(f"  [WARN] Modèle {model_name} introuvable → SHAP ignoré.")
            continue

        print(f"  → Calcul SHAP pour {model_name}...")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Graphique 1 : Importance globale (bar)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        plt.title(f"Importance Absolue des Variables — {model_name}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plot1_path = config_macro.FIGURES_DIR / f"shap_importance_bar_{model_name.replace(' ', '_')}.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Graphique 2 : Impact directionnel (beeswarm)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"Directionnalité de l'effet — {model_name}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plot2_path = config_macro.FIGURES_DIR / f"shap_impact_beeswarm_{model_name.replace(' ', '_')}.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✅ Graphiques SHAP sauvegardés pour {model_name}")

    print(f"\n✅ Tous les graphiques SHAP sont dans : {config_macro.FIGURES_DIR}")