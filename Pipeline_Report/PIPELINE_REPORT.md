# MSE-2 House Price Prediction — Pipeline Report (v2, Final)

> **Environment**: `MSE_2_ANN` — pyenv virtualenv, Python 3.12.3
> **Dataset**: `06_house_prices.csv` — 750 rows × 7 columns
> **Models**: Linear Regression, Random Forest, Gradient Boosting *(syllabus scope)*

---

## Dataset Overview

| Column | Type | Range | Notes |
|---|---|---|---|
| `id` | int64 | 1–750 | Sequential row index — NOT a feature |
| `area_sqft` | int64 | 439–2864 | House area in square feet |
| `rooms` | int64 | 1–8 | Number of rooms |
| `age_years` | int64 | 0–60 | House age in years |
| `dist_city_km` | float64 | 0.5–20.5 | Distance from city centre (km) |
| `garage` | int64 | 0 or 1 | Binary: has garage (1) or not (0) |
| `price_usd` | float64 | 129,514–318,073 | **Target variable** |

---

## 9 Hidden Traps — Identified & Fixed

| # | Trap | Rows Affected | Impact | Resolution |
|---|---|---|---|---|
| **T1** | Extreme price outlier: $129,514 (z = −3.87) | id=558 | Pulls all model predictions down | Removed via \|z\| > 3 on price |
| **T2** | Physically tiny area < 500 sqft | ids 72, 74, 208 | Implausible data points skew area-price relationship | Removed (`area_sqft >= 500` filter) |
| **T3** | Isolated age spike: 60 yrs (only value > 52) | id=685 | Likely data-entry error; distorts age feature | Removed (`age_years <= 55` filter) |
| **T4** | Tiny area + high price: < 500 sqft but > $200k | ids 72, 208 | Physically inconsistent pricing | Caught and removed by T2 filter |
| **T5** | `id` is a sequential row index, not a feature | All rows | Using it as feature = data leakage | Dropped before any modelling |
| **T6** | `garage` stored as `int64` not `bool` | All rows | Treated as continuous when it's binary | Kept as 0/1; documented |
| **T7** | `dist_city_km` min = 0.5 km (possible floor) | All rows | May indicate clipping in data | Noted; no action taken |
| **T8** | Two rows with area z-score > 3 (2702, 2864 sqft) | ids 77, 125 | Could be flagged as outliers | Kept — prices proportionally higher ($269k, $316k) |
| **T9** | Original `main.py` uses IQR×3 filter | id=558 survives | $129k outlier is NOT removed by original code | Replaced with z-score ±3 |

> **Critical bug in original code**: `main.py` line 36 uses `Q1 - 3*IQR` as the lower bound.
> For this dataset: Q1=$216,360, IQR=$39,747 → lower bound = $216,360 − $119,241 = **$97,119**.
> The $129,514 outlier comfortably clears this threshold and is **silently retained**.

---

## Data Cleaning Summary

```
Raw dataset         : 750 rows
After price z-filter: 749 rows  (removed id=558)
After area/age filter: 745 rows  (removed ids 72, 74, 208, 685)
Final clean shape   : 745 rows × 6 columns
Price range (clean) : $154,060 – $318,073
```

---

## Feature Engineering

| Feature | Formula | Reason |
|---|---|---|
| `log_area` | `log1p(area_sqft)` | Area is right-skewed; log normalises distribution |
| `age_bucket` | `cut(age_years, [-1,10,25,55], labels=[0,1,2])` | Groups: new(<10yr), mid(10-25yr), old(>25yr) |

> **Dropped from v1**: `rooms_per_1000sqft` — F-regression score was ~0.9 (near zero vs 264 for `log_area`). It was noise.

**F-regression scores (train set, ranked):**

| Feature | F-score |
|---|---|
| `log_area` | 264.5 |
| `area_sqft` | 263.8 |
| `dist_city_km` | 111.9 |
| `rooms` | 87.6 |
| `age_years` | 79.9 |
| `age_bucket` | 69.8 |
| `garage` | 14.2 |

**Correlation with price_usd (clean data):**

| Feature | Correlation |
|---|---|
| `area_sqft` | **+0.592** |
| `rooms` | +0.381 |
| `garage` | +0.219 |
| `age_years` | −0.338 |
| `dist_city_km` | **−0.403** |

---

## Train / Test Split

- **Split**: 80% train / 20% test (random_state=42)
- **Train**: 596 rows
- **Test**: 149 rows
- **Scaler**: `StandardScaler` — fitted on train only to prevent data leakage into test

---

## Cross-Validation Results (5-fold, train set)

| Model | CV RMSE | ± Std | CV R² |
|---|---|---|---|
| **Linear Regression** | **15,681** | 1,602 | 0.6537 |
| Gradient Boosting | 17,734 | 908 | 0.5590 |
| Random Forest | 17,789 | 1,084 | 0.5574 |

---

## Hyperparameter Tuning (GridSearchCV, 5-fold)

**Random Forest:**
- Grid: `n_estimators` ∈ {100,200,300}, `max_depth` ∈ {None,10,20}, `min_samples_split` ∈ {2,5}
- Best params: `max_depth=10, min_samples_split=5, n_estimators=200`
- Best CV RMSE: **17,588**

**Gradient Boosting:**
- Grid: `n_estimators` ∈ {100,200,300}, `learning_rate` ∈ {0.05,0.1,0.2}, `max_depth` ∈ {3,4,5}
- Best params: `learning_rate=0.05, max_depth=4, n_estimators=100`
- Best CV RMSE: **16,904**

---

## Final Test-Set Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| 🏆 **Linear Regression** | **$16,248** | **$12,880** | **0.6749** |
| Gradient Boosting (tuned) | $17,540 | $13,763 | 0.6211 |
| Random Forest (tuned) | $18,200 | $14,100 | 0.5921 |
| Random Forest (default) | $18,437 | $14,274 | 0.5814 |
| Gradient Boosting (default) | $18,566 | $14,534 | 0.5755 |

> **Why Linear Regression wins**: The features have strong linear correlations with price (area ρ=0.59, dist ρ=−0.40). The dataset appears to have been generated from a linear function, which means tree models — despite being more flexible — are over-parameterised here and introduce variance without reducing bias.

---

## Linear Regression Coefficients

| Feature | Coefficient | Interpretation |
|---|---|---|
| `area_sqft` | +11,939 | Each extra sqft adds ~$12k (before log interaction) |
| `garage` | +6,291 | Garage presence adds ~$6.3k |
| `log_area` | +5,271 | Log transformation captures diminishing returns on area |
| `age_bucket` | +1,664 | Bucket encoding of age (positive due to ordering) |
| `rooms` | −1,444 | More rooms in same area → denser → slight price drop |
| `age_years` | −10,538 | Each year older → −$10.5k |
| `dist_city_km` | −11,049 | Each km further → −$11k |

---

## Sample Prediction

```
Input : area=1600 sqft | 4 rooms | age=10yr | dist=5km | garage=yes
Output: $264,109  (Linear Regression)
```

---

## Saved Artefacts

| File | Description |
|---|---|
| `distributions.png` | Feature histograms after cleaning |
| `boxplots.png` | Boxplots showing remaining spread |
| `correlation_matrix.png` | Lower-triangle heatmap |
| `predicted_vs_actual.png` | Scatter: actual vs predicted prices |
| `residuals.png` | Residual vs predicted + distribution + Q-Q plot |
| `feature_importances.png` | Linear coefficients (champion = LinearReg) |
| `model_comparison.png` | RMSE and R² bar chart for all 5 model variants |
| `champion_model.pkl` | Serialised champion model (joblib) |
| `scaler.pkl` | Fitted StandardScaler (joblib) |
