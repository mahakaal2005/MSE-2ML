# AI Interaction Log — MSE-2 Machine Learning Pipeline
**Date**: 21 April 2026
**Project**: House Price Prediction — End-to-End ML Pipeline
**Dataset**: `06_house_prices.csv`
**AI System**: Antigravity (Google DeepMind)
**Conversation ID**: 2d13221d-d9a9-4b13-afe3-db78f3759cac

---

## Session Overview

This log documents an AI-assisted machine learning workflow for MSE-2. The AI agent was used to:
1. Set up an isolated Python environment using pyenv
2. Discover system tool locations (pyenv, nvm, node)
3. Analyse a house prices dataset for hidden data quality issues
4. Build a complete ML pipeline from raw data to prediction
5. Self-review, identify and fix mistakes in the first draft
6. Produce a corrected, submission-ready pipeline

---

## Interaction 1 — Environment Setup Request

**User prompt**: Install HuggingFace skills pack and set up the project.

**AI actions**:
- Attempted `npx skills add huggingface/skills` — failed because `npx` was not on PATH
- Discovered that Node.js is managed by **nvm** (`~/.nvm`) and is not on the default PATH
- User confirmed skills installed via their own terminal

---

## Interaction 2 — Tool Discovery

**User prompt**: Find all custom tool locations; everything is installed in custom locations.

**AI actions**:
- Read `/home/mahakaal/.zshrc` to extract PATH configuration
- Discovered the following (not available in default PATH):

| Tool | Location | Init required |
|---|---|---|
| pyenv | `~/.pyenv/bin/pyenv` | `export PYENV_ROOT="$HOME/.pyenv" && eval "$(pyenv init -)"` |
| nvm | `~/.nvm/nvm.sh` | `source "$NVM_DIR/nvm.sh"` |
| Node.js | via nvm | v24.12.0 |
| npm / npx | via nvm | 11.6.2 |

- Saved this as a persistent Knowledge Item at:
  `/home/mahakaal/.gemini/antigravity/knowledge/mahakaal-system-config/`

---

## Interaction 3 — Python Environment Creation

**User prompt**: Create an isolated Python environment named `MSE_2_ANN` using pyenv.

**AI actions**:
```bash
# Command used (with correct pyenv prefix):
export PYENV_ROOT="$HOME/.pyenv" && export PATH="$PYENV_ROOT/bin:$PATH" && eval "$(pyenv init -)"
pyenv virtualenv 3.12.3 MSE_2_ANN
pyenv local MSE_2_ANN          # sets .python-version in project dir
```

- Created virtualenv `MSE_2_ANN` based on Python 3.12.3
- Set it as the local Python for `/home/mahakaal/Dev/College/machine_learning/MSE-2/`
- Installed packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `joblib`

---

## Interaction 4 — Dataset Audit (Hidden Trap Discovery)

**User prompt**: Identify all problems ("hidden traps") in the dataset.

**AI actions**:
Ran a diagnostic Python script against `06_house_prices.csv` and found:

### Raw Dataset Stats
- Shape: 750 rows × 7 columns
- No missing values
- No duplicate rows or IDs

### Traps Found

| Trap | Finding |
|---|---|
| T1 | `id=558`: price=$129,514 — z-score = −3.87. Far below the 1st percentile ($173k). |
| T2 | 3 rows with `area_sqft < 500`: ids 72 (461 sqft), 74 (448 sqft), 208 (439 sqft) |
| T3 | `id=685`: `age_years=60` — the only value above 52; isolated spike |
| T4 | ids 72, 208: area < 500 sqft yet priced above $200k — physically inconsistent |
| T5 | `id` column: correlation with price = −0.049 — meaningless sequential index |
| T6 | `garage` stored as `int64` but only values are 0 and 1 — binary flag |
| T7 | `dist_city_km` minimum = 0.5 — possible floor/clipping artifact |
| T8 | ids 77 (2702 sqft) and 125 (2864 sqft) have area z-score > 3 — but kept as valid (prices proportionally high) |
| T9 | Original `main.py` uses IQR×3 outlier filter which is too lenient — id=558 is NOT removed by it |

---

## Interaction 5 — Pipeline v1 (First Draft)

**AI actions**: Wrote `pipeline.py` covering:
- 12 steps from data loading to sample prediction
- 6 models including ANN (MLPRegressor), Ridge, Lasso, Linear Regression, Random Forest, Gradient Boosting
- Feature engineering: log_area, rooms_per_1000sqft, age_bucket, dist_bucket

**Results (v1)**:
- Champion: Linear Regression (RMSE=$16,861, R²=0.7075)
- All plots and model outputs generated

---

## Interaction 6 — Self-Review (User Requested)

**User prompt**: "Pause — review your previous answer for mistakes, missing steps, unsupported assumptions and invented details."

**AI self-review findings**:

| Issue | Description |
|---|---|
| Bug 1 | T2 cleaning threshold was `area >= 450` but trap said `area < 500`. Row 72 (461 sqft) was flagged as bad but not removed. |
| Bug 2 | GridSearchCV condition for GBR tuning was always-False when best_model=Ridge. GBR was silently never tuned. |
| Bug 3 | `feature_importances.png` reported as "saved" in the summary but Linear Regression has no `.feature_importances_` — it was never actually generated. |
| Bug 4 | `rooms_per_1000sqft` added as a feature but had F-score of 0.9 (noise) — should have been dropped. |
| Invented | T7 described as "data generation clipping artifact" — this is unsupported; only the min value was known. |
| Scope | ANN (MLPRegressor), Ridge, and Lasso are not in the course syllabus — should be removed. |
| Missing | No model persistence (`joblib.dump`), no Q-Q plot for residual normality. |

**Confidence rating of v1**: 5/10

---

## Interaction 7 — Pipeline v2 (Corrected)

**AI actions**: Rewrote `pipeline.py` fixing all identified issues:

### Changes from v1 to v2

| Item | v1 | v2 |
|---|---|---|
| T2 filter | `area >= 450` (missed row 72) | `area >= 500` (all 3 rows removed) |
| GridSearch | Conditional — GBR skipped when Ridge won CV | Both RF and GBR tuned unconditionally |
| Feature importances | Claimed to save PNG for LinReg champion — incorrect | Falls back to coefficient plot for linear models |
| Noise feature | `rooms_per_1000sqft` kept (F=0.9) | Dropped |
| T7 description | "data generation clipping artifact" | "possible floor in data" |
| Models | 6 models incl. ANN, Ridge, Lasso | 3 models: LinReg, RF, GBR only |
| Model saving | Not implemented | `joblib.dump` for champion + scaler |
| Residuals | 2-panel plot | 3-panel: residuals + distribution + Q-Q plot |

### v2 Cleaning Results
```
Raw:                750 rows
After price z=3:    749 rows  (removed id=558, $129,514)
After area/age:     745 rows  (removed ids 72, 74, 208, 685)
```

### v2 Final Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Linear Regression 🏆** | **$16,248** | **$12,880** | **0.6749** |
| GBR (tuned) | $17,540 | $13,763 | 0.6211 |
| RF (tuned) | $18,200 | $14,100 | 0.5921 |
| RF (default) | $18,437 | $14,274 | 0.5814 |
| GBR (default) | $18,566 | $14,534 | 0.5755 |

**Sample prediction**: 1600 sqft, 4 rooms, age=10yr, dist=5km, garage=yes → **$264,109**

**Confidence rating of v2**: 8/10

---

## Files Generated

| File | Location |
|---|---|
| `pipeline.py` | `/home/mahakaal/Dev/College/machine_learning/MSE-2/` |
| `distributions.png` | Same directory |
| `boxplots.png` | Same directory |
| `correlation_matrix.png` | Same directory |
| `predicted_vs_actual.png` | Same directory |
| `residuals.png` | Same directory |
| `feature_importances.png` | Same directory (linear coefficients) |
| `model_comparison.png` | Same directory |
| `champion_model.pkl` | Same directory |
| `scaler.pkl` | Same directory |

---

## AI Tools & Packages Used

| Tool | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations, z-score |
| `scipy.stats` | Z-score, Q-Q plot |
| `matplotlib` / `seaborn` | All visualisations |
| `scikit-learn` | Models, scaling, cross-validation, GridSearch |
| `joblib` | Model serialisation |
| pyenv + virtualenv | Isolated Python environment (MSE_2_ANN) |
| HuggingFace skills pack | Installed via `npx skills add huggingface/skills` |
