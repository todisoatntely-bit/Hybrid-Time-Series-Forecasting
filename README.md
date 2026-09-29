```markdown
# Hybrid Time-Series Forecasting — MGA/USD

**Master’s thesis (Applied Mathematics & Optimization, University of Antananarivo).**

Hybrid econometrics + machine learning for the Malagasy Ariary / US Dollar exchange rate, with a strictly causal walk-forward protocol and an economic evaluation (hit rate, net PnL, Sharpe).

**Reference framework:** Ince & Trafalis (2006) — parametric input selection (ARIMA / cointegration), then non-parametric estimation (SVR, neural nets). 

**Code:** `Forecasting_MGA_USD_Hybrid/` (daily, univariate) · `Forecasting_MGA_USD_Macro/` (monthly, multivariate + SHAP).

---

## Why two pillars?

| Pillar | Frequency | Design | Main question |
| :--- | :--- | :--- | :--- |
| **Volet 1** | Daily (~2,200 obs., 252-day OOS) | Univariate | Can we beat a random walk economically, not only in RMSE? |
| **Volet 2** | Monthly (2019–2026, ~12-month OOS) | Multivariate (Brent, CPI, policy rate, FX reserves) | Which macro drivers matter, and how robust is that reading? |

Daily Brent was not forced into Volet 1. Engle–Granger on MGA/USD vs Brent: $t = -1.95$ (5% critical value $\approx -3.34$) → no cointegration. Madagascar is a net oil importer, unlike typical commodity-currency cases. Inputs stay parsimonious, as in Ince & Trafalis.

Johansen (VAR(2)) on the monthly panel: no cointegration at 5%. Models use lagged levels ($t-1$), not a VECM.

---

## Volet 1 — Daily pipeline (no leakage)

* **ARIMA lag selection** on the training window only.
* **Target** = daily difference ($\Delta y_t$); reconstruction ($\hat{y}_t = y_{t-1} + \widehat{\Delta y}_t$) with the observed lag-1 (no error accumulation).
* **Feature scaling + target scaling** (inverse-transform after each prediction).
* **Models:** SVR, MLP, XGBoost, LSTM (timesteps = 1: lags are already explicit features), plus a simple ensemble.
* **Expanding-window walk-forward** (252 steps). Light temporal hyperparameter search (SVR, XGBoost). Periodic XGBoost retrain.
* **Confidence filter:** trade only if $\vert{}\widehat{\Delta y}_t\vert{} \ge \tau$ (grid $\tau \in \{0, 2, 5, 10\}$ Ariary). Transaction cost 0.1% on position changes.

### Out-of-sample results (best threshold per model)

| Model | RMSE | Hit Rate | Net profit | Sharpe | Best ($\tau$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | 20.25 | 57.7% | +16.4% | 1.56 | 5 |
| **SVR** | 20.98 | 56.0% | +15.3% | 1.34 | 2 |
| **LSTM** | 20.06 | 58.1% | +13.1% | 1.09 | 10 |
| **Ensemble** | 20.39 | 58.5% | +13.0% | 0.99 | 5 |
| **XGBoost** | 20.64 | 54.0% | +12.0% | 0.77 | 5 |
| **Random walk** | 21.88 | 0% | 0% | 0 | — |

> RMSE gains vs the random walk are modest (Meese–Rogoff still bites). Economic value appears after the confidence filter: without it, Sharpes stay close to 0.3–0.5.
> **Retained specification:** MLP + ($\tau = 5$) Ariary.

---

## Volet 2 — Monthly macro + SHAP

* **Models:** Ridge, SVR, Random Forest, XGBoost. Benchmarks: random walk (level). Diebold–Mariano reported, but $n \approx 12$ — p-values and hit rates are fragile.
* Predictive edge vs RW is small (XGBoost only slightly better on MSE).
* **SHAP** (tree models, test set): importance is not unique across models (CPI lag vs Brent lag / FX lag). Policy rate and reserve coverage (months of imports) are consistently the least influential.
* **Limitation** (discussed in the thesis): the 4,550 → 4,150 drop (early 2026) is not anticipated by any model.

*Note: Volet 2 is an interpretability / policy-reading exercise, not a claim of strong monthly forecast skill.*

---

## Methodological claims (what this repo actually does)

* **Strict temporal causality** (walk-forward, train-only lag selection and scalers).
* **Dual evaluation:** statistical (RMSE, MAE, DM) and economic (directional accuracy, net PnL, Sharpe).
* **Honest negative results:** no daily cointegration with Brent; modest RMSE edge; monthly sample too small for strong inference.

*What it does not claim:* high-frequency trading, a “combinatorial SHAP engine”, or a proven Brent → Ariary causal channel at daily frequency.

---

## Repository structure

```text
Hybrid-Time-Series-Forecasting/
├── Forecasting_MGA_USD_Hybrid/     # daily univariate pipeline
│   ├── data/
│   ├── src/                        # ARIMA lags, prep, training, walk-forward, metrics
│   └── main.py
└── Forecasting_MGA_USD_Macro/      # monthly multivariate + SHAP
    ├── data/
    ├── src/
    └── main_macro.py

```

---

## Policy relevance (LICs / energy importers)

Useful framing, not a World Bank product:

* Energy-price shocks and FX pressure in a net oil-importing economy.
* Why a variable can be economically plausible and still fail a cointegration test at daily frequency.
* Risk-adjusted reading of a forecast (Sharpe / costs), not RMSE alone.

---

## How to run

```bash
# Volet 1
cd Forecasting_MGA_USD_Hybrid
python main.py

# Volet 2
cd Forecasting_MGA_USD_Macro
python main_macro.py

```

*Requirements: Python 3.10+, pandas, scikit-learn, xgboost, tensorflow/keras, shap, statsmodels, joblib.*

---

## Citation

> **Randriambolamanana, T. T. (2026).** *Prévision hybride du taux de change MGA/USD : approche économétrie–machine learning, validation walk-forward et interprétation économique.* Master 2, Université d’Antananarivo.

> **Ince, H. & Trafalis, T. B. (2006).** A hybrid model for exchange rate prediction. *Decision Support Systems*, 42(2), 1054–1062.

```

```