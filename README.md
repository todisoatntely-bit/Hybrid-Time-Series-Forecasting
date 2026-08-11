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
├── 📂 Forecasting_MGA_USD_Hybrid/     # HIGH-FREQUENCY / DAILY TRADING MODEL
│   ├── 📂 data/                        # 2,200+ daily exchange rate observations
│   ├── 📂 src/                         # Optimization pipelines (SVR, XGBoost, LSTM)
│   └── main.py                         # Walk-Forward Cross-Validation
│
└── 📂 Forecasting_MGA_USD_Macro/      # MULTIVARIATE / MACROECONOMIC MODEL
    ├── 📂 data/                        # Monthly macro panel (Brent, CPI, Policy Rate, Reserves)
    ├── 📂 src/                         # Ensemble models & Combinatorial SHAP engine
    └── main_macro.py                   # Macroeconomic pipeline execution