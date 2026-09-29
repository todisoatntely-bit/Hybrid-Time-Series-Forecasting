# 📈 Quantitative FX Forecasting: Algorithmic Optimization & Macro-Financial Analytics

A quantitative research project (Master 2 - Applied Mathematics & Optimization) focused on forecasting non-stationary financial time series (USD/MGA exchange rate) through hybrid machine learning, deep learning, and game-theoretic interpretability.

---

## 🌐 Alignment with World Bank Operational Priorities
This project directly addresses core quantitative challenges in Low-Income Countries (LICs):
* **Exogenous Shock Modeling:** Quantifying international energy price transmission (Brent crude) on local currency depreciation via SHAP values.
* **Central Bank Policy Analytics:** Evaluating inflation transmission, foreign exchange reserve coverage, and policy rate dynamics.
* **Quantitative Risk Management:** SVR quadratic optimization and deep learning architectures evaluated on risk-adjusted metrics (Sharpe Ratio).

---
## 📌 Repository Structure

```text
Hybrid-Time-Series-Forecasting/
│
├── Forecasting_MGA_USD_Hybrid/        # HIGH-FREQUENCY / DAILY TRADING MODEL
│   ├── data/                          # Dataset (2,200+ daily exchange rate observations)
│   ├── models/                        # Saved trained models
│   ├── outputs/                       # Generated results and visualizations
│   │   ├── figures/                   # Comparative plots (e.g., forecast_comparison.png)
│   │   ├── log/                       # Execution logs
│   │   ├── results/                   # CSV files (predictions, metrics, walk-forward results)
│   │   └── tables/                    # Tabular summaries
│   ├── src/                           # Source code (optimization pipelines, preprocessing)
│   ├── config.py                      # Configuration and hyperparameters
│   ├── main.py                        # Main execution script (Walk-Forward Cross-Validation)
│   └── requirements.txt               # Python project dependencies
│
└── Forecasting_MGA_USD_Macro/         # MULTIVARIATE / MACROECONOMIC MODEL
    ├── data/                          # Monthly macro panel (Brent, CPI, Policy Rate, Reserves)
    ├── models/                        # Saved trained macro models
    ├── outputs/                       # Generated results and visualizations
    │   ├── figures/                   # Comparative plots and SHAP visualizations
    │   └── tables/                    # Tabular summaries
    ├── src/                           # Source code (Macro pipeline)
    │   ├── __init__.py                # Package initialization
    │   ├── step01_macro_data_preparation.py # Data cleaning and feature engineering
    │   ├── step02_model_training.py   # Model training and optimization
    │   ├── step03_evaluation.py       # Performance evaluation metrics
    │   └── step04_economic_interpretation.py # Combinatorial SHAP engine
    ├── config_macro.py                # Configuration and hyperparameters for macro model
    └── main_macro.py                  # Macroeconomic pipeline execution