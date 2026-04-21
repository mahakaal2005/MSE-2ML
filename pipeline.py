"""
MSE-2 House Price Prediction Pipeline (v2 — corrected)
=======================================================
Environment : MSE_2_ANN  (pyenv 3.12.3)
Dataset     : 06_house_prices.csv  (750 rows × 7 cols)
Models      : Linear Regression, Random Forest, Gradient Boosting
              (ANN / Ridge / Lasso excluded — not in syllabus)

HIDDEN TRAPS IDENTIFIED
────────────────────────
 T1. price_usd extreme outlier  — id=558: $129,514  (z = -3.87)
     └─ Original IQR×3 filter in main.py is too lenient and misses this.
 T2. Physically tiny area       — ids 72, 74, 208: area < 500 sqft
     └─ Flagged as likely bad data; all three removed.
 T3. Isolated age spike         — id=685: age=60, only row > 52 yrs
     └─ Isolated spike suggests data-entry error; removed.
 T4. Tiny area + high price     — ids 72, 208: area < 500 sqft, price > $200k
     └─ Physically inconsistent; caught by T2 filter.
 T5. 'id' is a row index        — corr(id, price) = -0.049 → no signal
     └─ Must be dropped before any modelling.
 T6. 'garage' dtype mismatch    — binary 0/1 stored as int64
     └─ Kept as-is; documented for awareness.
 T7. dist_city_km min = 0.5 km  — may indicate a floor in the data
     └─ No action taken; noted for transparency.
 T8. Two large-area z-score hits — ids 77 (2702 sqft) and 125 (2864 sqft)
     └─ Prices are proportionally higher ($269k, $316k); treated as valid.
 T9. Original code bug           — main.py IQR×3 misses T1; replaced with z-score.
"""

import warnings
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection   import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics           import mean_squared_error, r2_score, mean_absolute_error

SEED   = 42
CSV    = '06_house_prices.csv'

def banner(text):
    print(f"\n{'═'*62}\n  {text}\n{'═'*62}")

def trap(n, msg):
    print(f"  ⚠️  T{n}: {msg}")

def fix(msg):
    print(f"  ✅  {msg}")

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Load & Raw Inspection
# ══════════════════════════════════════════════════════════════════
banner("STEP 1 — Load & Raw Inspection")

df_raw = pd.read_csv(CSV)
print(f"Shape      : {df_raw.shape}")
print(f"Columns    : {df_raw.columns.tolist()}")
print(f"\nDtypes:\n{df_raw.dtypes}")
print(f"\nNull values:\n{df_raw.isnull().sum()}")
print(f"\nDescriptive stats:\n{df_raw.describe().to_string()}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Hidden Trap Detection
# ══════════════════════════════════════════════════════════════════
banner("STEP 2 — Hidden Trap Detection")

df = df_raw.copy()

# T5 — 'id' as pseudo-feature
trap(5, "'id' is a sequential index — corr(id, price_usd) = "
        f"{df['id'].corr(df['price_usd']):.4f}")

# T6 — 'garage' binary but int64
trap(6, f"'garage' dtype=int64 but only values are {sorted(df['garage'].unique())} "
        "(binary flag, not continuous)")

# T1 — price extreme outlier
price_z = np.abs(stats.zscore(df['price_usd']))
t1_mask = price_z > 3
trap(1, f"Price z-score >3 — {t1_mask.sum()} row(s):")
print(df[t1_mask][['id','area_sqft','rooms','age_years','dist_city_km','price_usd']].to_string(index=False))

# T2 — area < 500 sqft
t2_mask = df['area_sqft'] < 500
trap(2, f"area_sqft < 500 — {t2_mask.sum()} row(s):")
print(df[t2_mask][['id','area_sqft','rooms','age_years','price_usd']].to_string(index=False))

# T3 — age spike (only row above 55)
t3_mask = df['age_years'] > 55
trap(3, f"age_years > 55 — {t3_mask.sum()} row(s):")
print(df[t3_mask][['id','area_sqft','rooms','age_years','price_usd']].to_string(index=False))

# T4 — tiny area + high price (subset of T2)
t4_mask = (df['area_sqft'] < 500) & (df['price_usd'] > 200_000)
trap(4, f"area<500 AND price>$200k — {t4_mask.sum()} row(s) (caught by T2 filter):")
print(df[t4_mask][['id','area_sqft','price_usd']].to_string(index=False))

# T7 — dist floor
trap(7, f"dist_city_km minimum = {df['dist_city_km'].min()} km — possible floor in data")

# T8 — large area (z>3) — treated as valid
area_z = np.abs(stats.zscore(df['area_sqft']))
t8_mask = area_z > 3
trap(8, f"area_sqft z-score >3 — {t8_mask.sum()} row(s) — kept (prices proportionally higher):")
print(df[t8_mask][['id','area_sqft','rooms','price_usd']].to_string(index=False))

# T9 — original code bug
trap(9, "main.py uses IQR×3 filter which is too lenient — id=558 ($129,514) survives it. "
        "Switched to z-score ±3 here.")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Data Cleaning
# ══════════════════════════════════════════════════════════════════
banner("STEP 3 — Data Cleaning")

df = df_raw.copy()

# Fix T5
df.drop(columns=['id'], inplace=True)
fix("Dropped 'id' column")

# Fix T1 — price z-score outlier
price_z = np.abs(stats.zscore(df['price_usd']))
n_before = len(df)
df = df[price_z <= 3].copy()
fix(f"Removed {n_before - len(df)} price outlier(s) via |z|>3  → {df.shape[0]} rows")

# Fix T2 + T3 — area < 500 OR age > 55
n_before = len(df)
df = df[(df['area_sqft'] >= 500) & (df['age_years'] <= 55)].copy()
fix(f"Removed {n_before - len(df)} row(s): area<500 OR age>55  → {df.shape[0]} rows")

print(f"\nFinal clean shape : {df.shape}")
print(f"Price range       : ${df['price_usd'].min():,.0f} – ${df['price_usd'].max():,.0f}")
print(f"\nDescriptive stats (clean):\n{df.describe().to_string()}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — EDA & Visualisations
# ══════════════════════════════════════════════════════════════════
banner("STEP 4 — EDA & Visualisations")

# 4a — Distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
feature_cols = [c for c in df.columns if c != 'price_usd'] + ['price_usd']
for ax, col in zip(axes.flatten(), feature_cols):
    ax.hist(df[col], bins=30, edgecolor='white', color='steelblue', alpha=0.85)
    ax.set_title(col, fontsize=11, fontweight='bold')
    ax.set_xlabel(col); ax.set_ylabel('Count')
plt.suptitle('Feature Distributions (after cleaning)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('distributions.png', dpi=120); plt.close()
print("Saved: distributions.png")

# 4b — Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.flatten(), feature_cols):
    ax.boxplot(df[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='darkred'),
               medianprops=dict(color='white', linewidth=2))
    ax.set_title(col, fontsize=11, fontweight='bold')
plt.suptitle('Boxplots (after cleaning)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('boxplots.png', dpi=120); plt.close()
print("Saved: boxplots.png")

# 4c — Correlation heatmap
corr = df.corr()
plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            mask=mask, vmin=-1, vmax=1,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=120); plt.close()
print("Saved: correlation_matrix.png")

print(f"\nCorrelation with price_usd:\n"
      f"{corr['price_usd'].sort_values(ascending=False).to_string()}")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Feature Engineering
# ══════════════════════════════════════════════════════════════════
banner("STEP 5 — Feature Engineering")

# Log-transform area (reduces right skew)
df['log_area'] = np.log1p(df['area_sqft'])
fix("Added log_area = log1p(area_sqft)")

# Age bucket
df['age_bucket'] = pd.cut(df['age_years'], bins=[-1, 10, 25, 55],
                           labels=[0, 1, 2]).astype(int)
fix("Added age_bucket: 0=new(<10yr)  1=mid(10-25)  2=old(>25)")

# NOTE: rooms_per_1000sqft was in v1 but had F-score ~0.9 (near-zero) — dropped.

print(f"\nShape after feature engineering: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ══════════════════════════════════════════════════════════════════
# STEP 6 — Train / Test Split & Scaling
# ══════════════════════════════════════════════════════════════════
banner("STEP 6 — Train / Test Split & Scaling")

X = df.drop(columns=['price_usd'])
y = df['price_usd']

# 80 / 20 split (simpler — no separate val set needed at this scale)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED)

print(f"Train : {X_train.shape[0]} rows")
print(f"Test  : {X_test.shape[0]} rows")
print(f"Features: {X_train.columns.tolist()}")

# StandardScaler — fit ONLY on train to prevent data leakage
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("\nStandardScaler fit on train only (no leakage into test)")

# ══════════════════════════════════════════════════════════════════
# STEP 7 — Model Training & 5-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════════
banner("STEP 7 — Model Training & 5-Fold Cross-Validation")

# Models in scope (syllabus): Linear Regression, Random Forest, Gradient Boosting
models = {
    'Linear Regression' : (LinearRegression(),                                           True),
    'Random Forest'     : (RandomForestRegressor(n_estimators=200, random_state=SEED),  False),
    'Gradient Boosting' : (GradientBoostingRegressor(n_estimators=200,
                            learning_rate=0.1, max_depth=4, random_state=SEED),         False),
}

cv_results = {}
print(f"\n{'Model':<22} {'CV RMSE':>12}  {'±':>8}  {'CV R²':>8}")
print("-" * 56)
for name, (model, use_scaled) in models.items():
    X_cv = X_train_sc if use_scaled else X_train.values
    rmse_scores = cross_val_score(model, X_cv, y_train.values,
                                  cv=5, scoring='neg_root_mean_squared_error')
    r2_scores   = cross_val_score(model, X_cv, y_train.values,
                                  cv=5, scoring='r2')
    cv_results[name] = -rmse_scores.mean()
    print(f"{name:<22} {-rmse_scores.mean():>12,.0f}  "
          f"{rmse_scores.std():>8,.0f}  {r2_scores.mean():>8.4f}")

best_cv_name = min(cv_results, key=cv_results.get)
print(f"\n→ Best by CV RMSE: {best_cv_name}  ({cv_results[best_cv_name]:,.0f})")

# ══════════════════════════════════════════════════════════════════
# STEP 8 — Hyperparameter Tuning (all three models)
# ══════════════════════════════════════════════════════════════════
banner("STEP 8 — Hyperparameter Tuning (GridSearchCV, 5-fold)")

# Linear Regression has no meaningful hyperparameters to tune — skip.
# Tune RF and GBR unconditionally.

param_grid_rf = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=SEED),
                       param_grid_rf, cv=5,
                       scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_rf.fit(X_train.values, y_train.values)
print(f"RF  best params : {grid_rf.best_params_}")
print(f"RF  best CV RMSE: {-grid_rf.best_score_:,.0f}")

param_grid_gb = {
    'n_estimators'  : [100, 200, 300],
    'learning_rate' : [0.05, 0.1, 0.2],
    'max_depth'     : [3, 4, 5],
}
grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=SEED),
                       param_grid_gb, cv=5,
                       scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_gb.fit(X_train.values, y_train.values)
print(f"\nGBR best params : {grid_gb.best_params_}")
print(f"GBR best CV RMSE: {-grid_gb.best_score_:,.0f}")

# ══════════════════════════════════════════════════════════════════
# STEP 9 — Final Evaluation on Test Set (all models)
# ══════════════════════════════════════════════════════════════════
banner("STEP 9 — Final Test-Set Evaluation")

# Fit all models on full training set, then evaluate on held-out test set
final_models = {
    'Linear Regression'      : (LinearRegression(),         True),
    'Random Forest (default)': (RandomForestRegressor(
                                    n_estimators=200, random_state=SEED), False),
    'Random Forest (tuned)'  : (grid_rf.best_estimator_,   False),
    'Gradient Boosting (default)': (GradientBoostingRegressor(
                                    n_estimators=200, learning_rate=0.1,
                                    max_depth=4, random_state=SEED), False),
    'Gradient Boosting (tuned)'  : (grid_gb.best_estimator_, False),
}

results = []
print(f"\n{'Model':<32} {'RMSE':>10}  {'MAE':>10}  {'R²':>8}")
print("-" * 66)
for name, (model, use_scaled) in final_models.items():
    X_fit  = X_train_sc if use_scaled else X_train.values
    X_eval = X_test_sc  if use_scaled else X_test.values
    model.fit(X_fit, y_train.values)
    preds  = model.predict(X_eval)
    rmse   = np.sqrt(mean_squared_error(y_test.values, preds))
    mae    = mean_absolute_error(y_test.values, preds)
    r2     = r2_score(y_test.values, preds)
    results.append({'name': name, 'model': model, 'preds': preds,
                    'rmse': rmse, 'mae': mae, 'r2': r2,
                    'scaled': use_scaled})
    print(f"{name:<32} {rmse:>10,.0f}  {mae:>10,.0f}  {r2:>8.4f}")

champion = min(results, key=lambda r: r['rmse'])
print(f"\n🏆 Champion : {champion['name']}")
print(f"   RMSE     : ${champion['rmse']:,.2f}")
print(f"   MAE      : ${champion['mae']:,.2f}")
print(f"   R²       : {champion['r2']:.4f}")

# ══════════════════════════════════════════════════════════════════
# STEP 10 — Visualisations
# ══════════════════════════════════════════════════════════════════
banner("STEP 10 — Visualisations")

y_pred = champion['preds']

# 10a — Predicted vs Actual
plt.figure(figsize=(8, 7))
plt.scatter(y_test, y_pred, alpha=0.55, color='steelblue',
            edgecolors='white', linewidths=0.4)
lims = [min(y_test.min(), y_pred.min()) - 5000,
        max(y_test.max(), y_pred.max()) + 5000]
plt.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
plt.xlabel('Actual Price (USD)'); plt.ylabel('Predicted Price (USD)')
plt.title(f'Predicted vs Actual — {champion["name"]}',
          fontsize=13, fontweight='bold')
plt.legend(); plt.tight_layout()
plt.savefig('predicted_vs_actual.png', dpi=120); plt.close()
print("Saved: predicted_vs_actual.png")

# 10b — Residuals
residuals = y_test.values - y_pred
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

ax1.scatter(y_pred, residuals, alpha=0.5, color='coral')
ax1.axhline(0, color='black', lw=1.2, ls='--')
ax1.set_xlabel('Predicted Price'); ax1.set_ylabel('Residual')
ax1.set_title('Residuals vs Predicted')

ax2.hist(residuals, bins=35, edgecolor='white', color='coral', alpha=0.85)
ax2.set_xlabel('Residual'); ax2.set_ylabel('Count')
ax2.set_title('Residual Distribution')

# Q-Q plot
stats.probplot(residuals, plot=ax3)
ax3.set_title('Q-Q Plot (residuals vs normal)')

plt.suptitle(f'Residual Analysis — {champion["name"]}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('residuals.png', dpi=120); plt.close()
print("Saved: residuals.png")

# 10c — Feature importances (only for tree-based models)
if hasattr(champion['model'], 'feature_importances_'):
    imp = pd.Series(champion['model'].feature_importances_,
                    index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#e07b54' if v == imp.max() else 'steelblue' for v in imp]
    imp.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title(f'Feature Importances — {champion["name"]}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=120); plt.close()
    print("Saved: feature_importances.png")
    print(f"\nFeature Importances:\n"
          f"{imp.sort_values(ascending=False).to_string()}")
else:
    print(f"(Champion is {champion['name']} — no feature importances. "
          f"Showing LinearRegression coefficients instead.)")
    coef = pd.Series(champion['model'].coef_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    coef.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(f'Coefficients — {champion["name"]}',
                 fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    plt.tight_layout()
    plt.savefig('feature_importances.png', dpi=120); plt.close()
    print("Saved: feature_importances.png  (linear coefficients)")
    print(f"\nCoefficients:\n{coef.sort_values(ascending=False).to_string()}")

# 10d — Model comparison bar chart
names  = [r['name'] for r in results]
rmses  = [r['rmse'] for r in results]
r2s    = [r['r2']   for r in results]
colors = ['#f4845f' if r['name'] == champion['name'] else '#4a9eda'
          for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
bars = ax1.barh(names, rmses, color=colors, edgecolor='white')
ax1.set_xlabel('RMSE (USD)')
ax1.set_title('Model Comparison — RMSE (lower is better)', fontweight='bold')
ax1.bar_label(bars, fmt='${:,.0f}', padding=3, fontsize=8)
bars2 = ax2.barh(names, r2s, color=colors, edgecolor='white')
ax2.set_xlabel('R²')
ax2.set_title('Model Comparison — R² (higher is better)', fontweight='bold')
ax2.bar_label(bars2, fmt='{:.4f}', padding=3, fontsize=8)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120); plt.close()
print("Saved: model_comparison.png")

# ══════════════════════════════════════════════════════════════════
# STEP 11 — Save Champion Model
# ══════════════════════════════════════════════════════════════════
banner("STEP 11 — Save Champion Model & Scaler")

joblib.dump(champion['model'], 'champion_model.pkl')
joblib.dump(scaler,            'scaler.pkl')
print(f"Saved: champion_model.pkl  ({champion['name']})")
print("Saved: scaler.pkl")

# ══════════════════════════════════════════════════════════════════
# STEP 12 — Predict on New Sample
# ══════════════════════════════════════════════════════════════════
banner("STEP 12 — Predict on New Sample")

# Must engineer features exactly as in training
sample = {
    'area_sqft'   : 1600,
    'rooms'       : 4,
    'age_years'   : 10,
    'dist_city_km': 5.0,
    'garage'      : 1,
    'log_area'    : np.log1p(1600),
    'age_bucket'  : 1,    # 10-25 yr bucket
}
sample_df = pd.DataFrame([sample])[X.columns]  # enforce column order

if champion['scaled']:
    sample_input = scaler.transform(sample_df)
else:
    sample_input = sample_df.values

pred_price = champion['model'].predict(sample_input)[0]
print(f"\nSample: area=1600 sqft | 4 rooms | age=10 yr | dist=5 km | garage=yes")
print(f"Predicted Price: ${pred_price:,.2f}  ({champion['name']})")

# Final summary
banner("PIPELINE COMPLETE")
saved = [
    'distributions.png', 'boxplots.png', 'correlation_matrix.png',
    'predicted_vs_actual.png', 'residuals.png',
    'feature_importances.png', 'model_comparison.png',
    'champion_model.pkl', 'scaler.pkl'
]
print("Saved artefacts:")
for f in saved:
    print(f"  📄  {f}")
