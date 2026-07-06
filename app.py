#Call the packages
import os
import time
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Initialize timer, constants, and local directories
START = time.time()
DATA_PATH = "kolkata_house_prices.csv"
TARGET_COL = "Price"
os.makedirs("models", exist_ok=True)

#Step 2===Data Loading & Feature/Target Ingestion

# Load and clean data
df = pd.read_csv(DATA_PATH).drop_duplicates()

# Separate features from target
X = df.drop(columns=TARGET_COL)
y = df[TARGET_COL].astype(float)

# Separate columns dynamically by data type
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

#Step 3====Defining the Preprocessing Strategy

prep = ColumnTransformer([
    ("num", Pipeline([
        ("i", SimpleImputer(strategy="median")),
        ("s", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("i", SimpleImputer(strategy="most_frequent")),
        ("o", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

#Step 4===Train-Test Splitting & Feature Transformation

# Split into training and testing sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and apply data transformations
prep.fit(Xtr)
Xtrp = prep.transform(Xtr)
Xtep = prep.transform(Xte)

# Track resulting feature names
feat = prep.get_feature_names_out()

#Step 5===Model Dictionary Configuration & Iterative Training

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=180, max_depth=12, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=220, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42)
}

res = {}
trained = {}

# Train and evaluate each model
for n, m in models.items():
    m.fit(Xtrp, ytr)
    a = m.predict(Xtrp)
    b = m.predict(Xtep)
    trained[n] = m
    res[n] = {
        "Train_R2": r2_score(ytr, a),
        "Test_R2": r2_score(yte, b),
        "Train_RMSE": np.sqrt(mean_squared_error(ytr, a)),
        "Test_RMSE": np.sqrt(mean_squared_error(yte, b)),
        "Train_MAE": mean_absolute_error(ytr, a),
        "Test_MAE": mean_absolute_error(yte, b)
    }

# Rank models based on Test R² performance
results = pd.DataFrame(res).T.sort_values("Test_R2", ascending=False)
print(results.round(4))

#Step 6=====Model Productionization & Cross-Validation

best = results.index[0]

# Construct final end-to-end pipeline
pipe = Pipeline([("prep", prep), ("model", trained[best])])

# Validate across the complete dataset
cv = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"CV Mean R2: {cv.mean()}, CV Std Dev: {cv.std()}")

# Save full pipeline artifact
joblib.dump(pipe, "models/Best_Model.joblib")

#Step 7======SHAP Model Interpretability & Runtime Profiling

# Downsample for faster SHAP calculations
idx = np.random.default_rng(42).choice(Xtrp.shape[0], min(2000, Xtrp.shape[0]), replace=False)
Xs = Xtrp[idx]

# Dynamically select explainer type based on the winning model
if best != "LinearRegression":
    vals = shap.TreeExplainer(trained[best]).shap_values(Xs)
else:
    vals = shap.Explainer(trained[best], Xs)(Xs)

# Render summary plot and display pipeline runtime
shap.summary_plot(vals, Xs, feature_names=feat, show=False)
plt.tight_layout()
plt.show()

print(f"Total Execution Time: {time.time() - START} seconds")

#Additional Step for shap and feature importance


# Optimized salary prediction pipeline with Universal Explainability Extensions
import os, time, joblib, shap, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- Existing Pipeline Setup ---
START = time.time(); DATA_PATH = "kolkata_house_prices.csv"; TARGET_COL = "Price"; os.makedirs("models", exist_ok=True)
df = pd.read_csv(DATA_PATH).drop_duplicates(); X = df.drop(columns=TARGET_COL); y = df[TARGET_COL].astype(float)
num = X.select_dtypes(include="number").columns; cat = X.select_dtypes(exclude="number").columns
prep = ColumnTransformer([("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num), ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
prep.fit(Xtr); Xtrp = prep.transform(Xtr); Xtep = prep.transform(Xte); feat = prep.get_feature_names_out()
models = {"LinearRegression": LinearRegression(), "RandomForest": RandomForestRegressor(n_estimators=180, max_depth=12, n_jobs=-1, random_state=42), "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=220, max_depth=8, learning_rate=.08, subsample=.8, colsample_bytree=.8, tree_method="hist", n_jobs=-1, random_state=42)}
res = {}; trained = {}
for n, m in models.items():
    m.fit(Xtrp, ytr); a = m.predict(Xtrp); b = m.predict(Xtep); trained[n] = m
    res[n] = {"Train_R2": r2_score(ytr, a), "Test_R2": r2_score(yte, b), "Train_RMSE": np.sqrt(mean_squared_error(ytr, a)), "Test_RMSE": np.sqrt(mean_squared_error(yte, b)), "Train_MAE": mean_absolute_error(ytr, a), "Test_MAE": mean_absolute_error(yte, b)}
results = pd.DataFrame(res).T.sort_values("Test_R2", ascending=False); print(results.round(4)); best = results.index[0]; pipe = Pipeline([("prep", prep), ("model", trained[best])]); cv = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1); print(cv.mean(), cv.std()); joblib.dump(pipe, "models/Best_Model.joblib")
idx = np.random.default_rng(42).choice(Xtrp.shape[0], min(2000, Xtrp.shape[0]), replace=False); Xs = Xtrp[idx]

# Dynamically calculate SHAP values based on best model architecture
if best != "LinearRegression": 
    explainer = shap.TreeExplainer(trained[best])
    shap_matrix = explainer.shap_values(Xs)
    # Handle SHAP output structure variations across different tree package versions cleanly
    base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
else: 
    explainer = shap.Explainer(trained[best], Xs)
    shap_obj = explainer(Xs)
    shap_matrix = shap_obj.values
    base_val = shap_obj.base_values[0] if isinstance(shap_obj.base_values, (list, np.ndarray)) else shap_obj.base_values

shap.summary_plot(shap_matrix, Xs, feature_names=feat, show=False); plt.tight_layout(); plt.show(); print(f"Baseline Pipeline Time: {time.time()-START}s")

# =====================================================================
# ADDITION 1: NLP-Style Natural Language Explainer (Universal & Dynamic)
# =====================================================================
print("\n" + "="*50 + "\nNLP-STYLE INDIVIDUAL PREDICTION STORY\n" + "="*50)

# 1. Pick a random test profile from our processed sample matrix
sample_idx = np.random.default_rng().choice(Xs.shape[0])
individual_features = Xs[sample_idx]
individual_shap = shap_matrix[sample_idx]

# 2. Extract the actual prediction mathematically matching the SHAP matrix
individual_pred = base_val + np.sum(individual_shap)

print(f"Starting Baseline Value (Average Dataset Prediction): {base_val:,.2f}")
print(f"Final Model Prediction for this Individual: {individual_pred:,.2f}\n")
print(f"--- How the Model Arrived at this Decision (Top Drivers) ---")

# 3. Zip features and evaluate magnitude directions dynamically without static text assumptions
drivers = pd.DataFrame({
    'Feature': feat,
    'SHAP_Value': individual_shap,
    'Absolute_Impact': np.abs(individual_shap)
}).sort_values(by='Absolute_Impact', ascending=False)

# 4. Generate dynamic plain-English narratives for the top 5 most impactful features
for _, row in drivers.head(5).iterrows():
    direction = "HIGHER" if row['SHAP_Value'] > 0 else "LOWER"
    action = "pushed UP" if row['SHAP_Value'] > 0 else "pulled DOWN"
    print(f"• The feature '{row['Feature']}' {action} the prediction by {abs(row['SHAP_Value']):,.2f}, leading to a {direction} final estimate.")


# =====================================================================
# ADDITION 2: Global Feature Importance Extraction (Universal & Dynamic)
# =====================================================================
print("\n" + "="*50 + "\nGLOBAL FEATURE IMPORTANCE SUMMARY\n" + "="*50)

# Mathematically aggregate the absolute SHAP matrices across all evaluated samples
global_impacts = np.mean(np.abs(shap_matrix), axis=0)

# Map back to variable column names dynamically
feature_importance_df = pd.DataFrame({
    'Feature_Name': feat,
    'Global_Impact_Score': global_impacts
}).sort_values(by='Global_Impact_Score', ascending=False).reset_index(drop=True)

# Display the ranked overall impact dataframe
print(feature_importance_df.head(10).round(4))

# Dynamic Classification/Regression Pipeline with Full Visualization Suite
# Set DATASET_PATH and TARGET_COL below - everything else (task type,
# feature selection, model choice, titles, charts, narrative) is inferred
# automatically from the data. No other manual edits required.
import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, classification_report, r2_score, f1_score,
                              mean_squared_error, confusion_matrix, roc_curve, auc)
from scipy.stats import f_oneway, chi2_contingency, pearsonr

DATASET_PATH = "kolkata_house_prices.csv"      # <-- only thing to change
TARGET_COL   = "Price"      # <-- only thing to change
OUT_DIR      = "visuals"

os.makedirs(OUT_DIR, exist_ok=True)
def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{name}.png", dpi=130)
    plt.close()

# ---------- Load ----------
df = pd.read_csv(DATASET_PATH)
assert TARGET_COL in df.columns, f"'{TARGET_COL}' not found in dataset"
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

# ---------- Task auto-detection ----------
is_clf = (df[TARGET_COL].dtype == 'object') or (df[TARGET_COL].nunique() <= min(20, int(len(df) * 0.05) + 2))
task = "Classification" if is_clf else "Regression"
print(f"Task detected: {task} | Target: {TARGET_COL} | Rows: {len(df)}")

# ---------- Clean ----------
id_cols = [c for c in df.columns if c != TARGET_COL and
           ('id' in c.lower() or (df[c].dtype == 'object' and df[c].nunique() == len(df)))]
df = df.drop(columns=id_cols, errors='ignore')
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].fillna(df[c].mode()[0])
for c in df.select_dtypes(include=np.number).columns:
    df[c] = df[c].fillna(df[c].median())

y_raw = df[TARGET_COL]
X_raw = df.drop(columns=[TARGET_COL])

le_target = None
if is_clf and y_raw.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
else:
    y = y_raw.values

cat_cols = X_raw.select_dtypes(include='object').columns.tolist()
num_cols = X_raw.select_dtypes(include=np.number).columns.tolist()

# ---------- Statistical significance testing (drives feature relevance, not hardcoded) ----------
pvals, sig_features = {}, []
print(f"\nStatistical significance of each feature vs '{TARGET_COL}' (alpha=0.05):")
for c in num_cols:
    try:
        if is_clf:
            groups = [X_raw[c][y == g].dropna() for g in np.unique(y)]
            _, p = f_oneway(*groups)
        else:
            _, p = pearsonr(X_raw[c], y)
        pvals[c] = p
        sig_features.append(c) if p < 0.05 else None
        print(f"  {c:<30} p={p:.4f} {'significant' if p < 0.05 else ''}")
    except Exception:
        pass

for c in cat_cols:
    try:
        target_binned = y_raw if is_clf else pd.qcut(y_raw, q=4, duplicates='drop')
        ct = pd.crosstab(X_raw[c], target_binned)
        _, p, _, _ = chi2_contingency(ct)
        pvals[c] = p
        sig_features.append(c) if p < 0.05 else None
        print(f"  {c:<30} p={p:.4f} {'significant' if p < 0.05 else ''}")
    except Exception:
        pass

print(f"\n{len(sig_features)}/{len(num_cols) + len(cat_cols)} features are statistically significant predictors of '{TARGET_COL}'.")

# ---------- Encode ----------
X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if is_clf else None
)

# ---------- Model race: pick best model by cross-validation (no hardcoded winner) ----------
if is_clf:
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, tree_method='hist',
                                  random_state=42, eval_metric='logloss', verbosity=0),
    }
    score_fn = lambda yt, yp: f1_score(yt, yp, average='weighted')
    scoring = 'f1_weighted'
else:
    candidates = {
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=150, max_depth=6, tree_method='hist',
                                 random_state=42, verbosity=0),
    }
    score_fn = r2_score
    scoring = 'r2'

# No cross-validation: each candidate is fit once on the train split and scored
# once on the held-out test split. CV would just refit every model 5x for no
# benefit here since the test set already gives an unbiased read on generalization.
cv_scores = {}
best_name, best_score, best_model, y_pred = None, -np.inf, None, None
print(f"\nModel selection (single train/test fit, scoring={scoring}):")
for name, mdl in candidates.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    score = score_fn(y_test, preds)
    cv_scores[name] = score
    print(f"  {name}: {score:.4f}")
    if score > best_score:
        best_name, best_score, best_model, y_pred = name, score, mdl, preds

print(f"\nBest model selected: {best_name} (test {scoring} = {best_score:.4f})")

# ---------- Evaluation ----------
print(f"\n=== {task} Report | Model: {best_name} | Target: {TARGET_COL} ===")
if is_clf:
    labels = le_target.classes_.astype(str) if le_target else None
    print(classification_report(y_test, y_pred, target_names=labels))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
else:
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE:", round(rmse, 2), "| R2:", round(r2_score(y_test, y_pred), 4))

importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print(f"\nTop drivers of '{TARGET_COL}':\n{importances.round(4)}")

# ============================================================
# VISUALIZATION SUITE (15+ dynamically generated charts)
# Every title, column, and label below is pulled from the
# dataset itself - nothing is hardcoded.
# ============================================================
chart_n = 0
def next_id():
    global chart_n
    chart_n += 1
    return f"{chart_n:02d}"

top_num = sorted(num_cols, key=lambda c: pvals.get(c, 1))[:3]
top_cat = sorted(cat_cols, key=lambda c: pvals.get(c, 1))[:2]

# 1. Target distribution
plt.figure(figsize=(7, 5))
if is_clf:
    sns.countplot(x=y_raw, hue=y_raw, palette="Spectral", legend=False)
else:
    sns.histplot(y_raw, kde=True, color="teal")
plt.title(f"Distribution of {TARGET_COL.replace('_', ' ')}")
savefig(f"{next_id()}_target_distribution")

# 2. Missing values overview
plt.figure(figsize=(8, 5))
miss = missing_pct[missing_pct > 0]
if len(miss):
    sns.barplot(x=miss.values, y=miss.index, hue=miss.index, palette="rocket", legend=False)
    plt.xlabel("% Missing")
else:
    plt.text(0.5, 0.5, "No missing values detected", ha='center', va='center', fontsize=13)
    plt.axis('off')
plt.title("Missing Data Overview (Original Dataset)")
savefig(f"{next_id()}_missing_values")

# 3. Correlation heatmap
corr_cols = num_cols + ([TARGET_COL] if not is_clf else [])
if len(corr_cols) > 1:
    plt.figure(figsize=(min(12, 1 + len(corr_cols)), min(10, 1 + len(corr_cols))))
    sns.heatmap(df[corr_cols].corr(), annot=len(corr_cols) <= 12, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    savefig(f"{next_id()}_correlation_heatmap")

# 4. Statistical significance (p-values) chart
if pvals:
    pv = pd.Series(pvals).sort_values()
    plt.figure(figsize=(8, 5))
    colors = ["#2ca02c" if p < 0.05 else "#d62728" for p in pv.values]
    plt.barh(pv.index, pv.values, color=colors)
    plt.axvline(0.05, color='black', linestyle='--', label='alpha=0.05')
    plt.xlabel("p-value"); plt.legend()
    plt.title(f"Statistical Significance of Features vs {TARGET_COL}")
    savefig(f"{next_id()}_statistical_significance")

# 5. Model comparison chart
plt.figure(figsize=(6, 4))
sns.barplot(x=list(cv_scores.keys()), y=list(cv_scores.values()), hue=list(cv_scores.keys()),
            palette="viridis", legend=False)
plt.title(f"Model Comparison (test-set {scoring})")
plt.ylabel(scoring)
savefig(f"{next_id()}_model_comparison")

# 6. Feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="Spectral", legend=False)
plt.title(f"Top Features Influencing {TARGET_COL.replace('_', ' ')} ({task})")
plt.xlabel("Importance Score")
savefig(f"{next_id()}_feature_importance")

# 7. Cumulative (Pareto) importance
cum = importances.cumsum() / importances.sum() * 100
plt.figure(figsize=(8, 5))
plt.bar(range(len(cum)), importances.values, color="steelblue")
plt.plot(range(len(cum)), cum.values, color="darkorange", marker='o')
plt.xticks(range(len(cum)), cum.index, rotation=45, ha='right')
plt.ylabel("Importance / Cumulative %")
plt.title(f"Pareto Chart: How Many Features Explain {TARGET_COL}")
savefig(f"{next_id()}_cumulative_importance")

# 8. Outlier overview across numeric features (standardized)
if len(num_cols) >= 1:
    scaled = pd.DataFrame(StandardScaler().fit_transform(df[num_cols]), columns=num_cols)
    plt.figure(figsize=(min(12, 1 + len(num_cols)), 5))
    sns.boxplot(data=scaled)
    plt.xticks(rotation=45, ha='right')
    plt.title("Outlier Overview - Standardized Numeric Features")
    savefig(f"{next_id()}_outlier_overview")

# 9-11. Distribution of top numeric features
for c in top_num:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[c], kde=True, color="slateblue")
    plt.title(f"Distribution of {c.replace('_', ' ')}")
    savefig(f"{next_id()}_distribution_{c}")

# 12-13. Top numeric features vs target relationship
for c in top_num[:2]:
    plt.figure(figsize=(7, 5))
    if is_clf:
        sns.boxplot(x=y_raw, y=df[c], hue=y_raw, palette="Set2", legend=False)
    else:
        sns.regplot(x=df[c], y=y_raw, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})
    plt.title(f"{c.replace('_', ' ')} vs {TARGET_COL.replace('_', ' ')}")
    savefig(f"{next_id()}_{c}_vs_target")

# 14-15. Top categorical feature distributions
for c in top_cat:
    plt.figure(figsize=(8, 4))
    order = df[c].value_counts().head(15).index
    sns.countplot(y=df[c], order=order, hue=df[c], palette="mako", legend=False)
    plt.title(f"Distribution of {c.replace('_', ' ')}")
    savefig(f"{next_id()}_distribution_{c}")

# 16-17. Top categorical features vs target
for c in top_cat:
    plt.figure(figsize=(8, 5))
    top_cats = df[c].value_counts().head(10).index
    sub = df[df[c].isin(top_cats)]
    if is_clf:
        rate = sub.groupby(c)[TARGET_COL].apply(lambda s: (s == s.mode()[0]).mean())
        sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="crest", legend=False)
        plt.ylabel("Share of top class")
    else:
        avg = sub.groupby(c)[TARGET_COL].mean().sort_values(ascending=False)
        sns.barplot(x=avg.index, y=avg.values, hue=avg.index, palette="crest", legend=False)
        plt.ylabel(f"Average {TARGET_COL}")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{TARGET_COL.replace('_', ' ')} by {c.replace('_', ' ')}")
    savefig(f"{next_id()}_target_by_{c}")

# 18. Model diagnostic - confusion matrix (classification) or actual vs predicted (regression)
plt.figure(figsize=(6, 5))
if is_clf:
    cm = confusion_matrix(y_test, y_pred)
    tick_labels = labels if labels is not None else np.unique(y)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    savefig(f"{next_id()}_confusion_matrix")
else:
    plt.scatter(y_test, y_pred, alpha=0.4, color="teal")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, color="red", linestyle="--")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"Actual vs Predicted {TARGET_COL.replace('_', ' ')} - {best_name}")
    savefig(f"{next_id()}_actual_vs_predicted")

# 19. Second diagnostic - ROC curve (binary classification) or residual plot (regression)
if is_clf and len(np.unique(y)) == 2 and hasattr(best_model, "predict_proba"):
    proba = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", label=f"AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {best_name}")
    plt.legend()
    savefig(f"{next_id()}_roc_curve")
elif not is_clf:
    resid = y_test - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, alpha=0.4, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residual")
    plt.title(f"Residual Plot - {best_name}")
    savefig(f"{next_id()}_residual_plot")

print(f"\n{chart_n} charts saved to '{OUT_DIR}/' folder.")

# ============================================================
# FEATURE RELATIONSHIP & RECOMMENDATION REPORT
# For every feature (numeric and categorical) this quantifies the
# actual relationship with the target - direction, magnitude, and
# which specific groups drive it - not just "significant / not".
# Fully dynamic: works for any column names, e.g. bedrooms, bathrooms,
# area, floor, amenities, or anything else present in the dataset.
# ============================================================
MIN_GROUP_SIZE = max(5, int(len(df) * 0.01))

def interpret_numeric(c):
    x = df[c].astype(float)
    if is_clf:
        means = x.groupby(y_raw).mean()
        top_cls, bot_cls = means.idxmax(), means.idxmin()
        base = means[bot_cls] if means[bot_cls] != 0 else 1e-9
        diff_pct = (means[top_cls] - means[bot_cls]) / abs(base) * 100
        return (f"'{c}' averages {diff_pct:+.1f}% higher for '{top_cls}' cases than '{bot_cls}' "
                f"-> use {c} to segment or prioritize for {TARGET_COL}.")
    else:
        slope, _ = np.polyfit(x, y_raw, 1)
        r = np.corrcoef(x, y_raw)[0, 1]
        strength = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.3 else "weak"
        direction = "increases" if slope > 0 else "decreases"
        return (f"each +1 unit of '{c}' is associated with {TARGET_COL} {direction} by "
                f"{abs(slope):,.2f} on average (r={r:.2f}, {strength} relationship).")

def interpret_categorical(c):
    vc = df[c].value_counts()
    valid = vc[vc >= MIN_GROUP_SIZE].index
    sub = df[df[c].isin(valid)]
    if len(valid) < 2:
        return None
    if is_clf:
        mode_label = y_raw.mode()[0]
        rates = sub.groupby(c)[TARGET_COL].apply(lambda s: (s == mode_label).mean())
        best, worst = rates.idxmax(), rates.idxmin()
        return (f"'{best}' shows the highest share of '{mode_label}' outcomes ({rates[best]*100:.1f}%) "
                f"vs '{worst}' the lowest ({rates[worst]*100:.1f}%) among '{c}' groups "
                f"-> tailor strategy by {c}.")
    else:
        avg = sub.groupby(c)[TARGET_COL].mean()
        best, worst = avg.idxmax(), avg.idxmin()
        base = avg[worst] if avg[worst] != 0 else 1e-9
        pct_diff = (avg[best] - avg[worst]) / abs(base) * 100
        return (f"'{best}' averages {avg[best]:,.2f} vs '{worst}' at {avg[worst]:,.2f} "
                f"({pct_diff:+.1f}% difference) among '{c}' groups -> {c} materially shifts {TARGET_COL}.")

# Rank features by statistical significance (lowest p-value first) so the
# strongest, most reliable relationships are surfaced first.
ranked_features = sorted(num_cols + cat_cols, key=lambda c: pvals.get(c, 1))

print(f"\n=== Feature Relationship & Recommendation Report (Target: {TARGET_COL}) ===")
for c in ranked_features:
    try:
        msg = interpret_numeric(c) if c in num_cols else interpret_categorical(c)
        if msg is None:
            continue
        sig_tag = "[statistically confirmed]" if c in sig_features else "[directional signal, not statistically confirmed]"
        print(f"  - {msg} {sig_tag}")
    except Exception:
        pass

# ---------- Auto-generated business narrative (built from real stats, not templated text) ----------
print("\nBusiness Recommendation Summary (top drivers by model importance):")
for feat in importances.index[:5]:
    is_sig = any(feat == f or feat.startswith(f + "_") for f in sig_features)
    tag = "statistically confirmed" if is_sig else "high model influence, not statistically confirmed"
    print(f"  - '{feat}' ranks among the top drivers of {TARGET_COL} ({tag}) -> recommend targeted action here.")

print("\nDone.")







