import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
import seaborn as sns
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _get_base_estimator(random_state=42, scale_weight=1.0):
    """Helper to create a base XGBoost estimator."""
    return XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=random_state, 
        scale_pos_weight=scale_weight
    )

# --- STEP 1: HYPERPARAMETER TUNING ---
def tune_hyperparameters(X, y_ret, t, param_grid, random_state=42):
    """
    Tunes hyperparameters for Control and Treated models independently using GridSearchCV.
    """
    print("\n--- 1. Tuning Hyperparameters ---")
    
    # Tune Control Model (Universe: Ignored)
    y_c = y_ret[t == 0]
    X_c = X[t == 0]
    ratio_c = np.sum(y_c == 0) / np.sum(y_c == 1)

    gs_c = GridSearchCV(
        _get_base_estimator(random_state, ratio_c), 
        param_grid, 
        cv=3, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    gs_c.fit(X_c, y_c)
    best_model_c = gs_c.best_estimator_
    print(f"Best Params (Control): {gs_c.best_params_}")

    # Tune Treated Model (Universe: Called)
    y_t = y_ret[t == 1]
    X_t = X[t == 1]
    ratio_t = np.sum(y_t == 0) / np.sum(y_t == 1)

    gs_t = GridSearchCV(
        _get_base_estimator(random_state, ratio_t), 
        param_grid, 
        cv=3, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    gs_t.fit(X_t, y_t)
    best_model_t = gs_t.best_estimator_
    print(f"Best Params (Treated): {gs_t.best_params_}")
    
    return best_model_c, best_model_t

# --- STEP 2: GENERATE OOF SCORES ---
def generate_oof_scores(X, y_ret, t, best_model_c, best_model_t, random_state=42):
    """
    Generates Out-of-Fold (OOF) Uplift scores using StratifiedKFold.
    """
    print("\n--- 2. Generating OOF Scores ---")
    
    oof_uplift = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Stratify by Treatment to ensure we can train T-Learner in every fold
    for train_idx, val_idx in skf.split(X, t):
        # Slice Data
        X_fold_train = X[train_idx]
        y_fold_train = y_ret[train_idx]
        t_fold_train = t[train_idx]

        X_fold_val = X[val_idx]

        # Train Temp Models
        m_c = clone(best_model_c)
        m_c.fit(X_fold_train[t_fold_train == 0], y_fold_train[t_fold_train == 0])
        
        m_t = clone(best_model_t)
        m_t.fit(X_fold_train[t_fold_train == 1], y_fold_train[t_fold_train == 1])

        # Predict Counterfactuals on Val
        p_c = m_c.predict_proba(X_fold_val)[:, 1]
        p_t = m_t.predict_proba(X_fold_val)[:, 1]

        # Uplift = P(Retention|Treated) - P(Retention|Control)
        oof_uplift[val_idx] = p_t - p_c
        
    return oof_uplift

# --- STEP 3: OPTIMIZE THRESHOLD ---
def determine_optimal_threshold(oof_uplift, y_churn, t):
    """
    Analyzes OOF scores to find the optimal targeting percentage that maximizes Net Churn Reduction.
    """
    print("\n--- 3. Determining Optimal Outreach % ---")

    # Combine OOF scores with actual Churn labels
    eval_df = pd.DataFrame({'uplift': oof_uplift, 'churn': y_churn, 'outreach': t})
    eval_df = eval_df.sort_values(by='uplift', ascending=False).reset_index(drop=True)

    x_axis = []
    net_impacts = []

    # Scan percentiles to find peak
    total_rows = len(eval_df)
    for k in range(5, 101, 5): # Check every 5%
        n = int(total_rows * (k/100))
        subset = eval_df.head(n)

        n_trt = subset['outreach'].sum()
        n_ctrl = n - n_trt

        if n_trt > 20 and n_ctrl > 20:
            rate_trt = subset[subset['outreach'] == 1]['churn'].mean()
            rate_ctrl = subset[subset['outreach'] == 0]['churn'].mean()
            # Net Saved = (Diff in Rate) * Total Population Targeted
            saved = (rate_ctrl - rate_trt) * n

            x_axis.append(k/100)
            net_impacts.append(saved)

    # Find Max
    if len(net_impacts) > 0:
        best_idx = np.argmax(net_impacts)
        optimal_pct = x_axis[best_idx]
        max_saved = net_impacts[best_idx]
    else:
        optimal_pct = 0.40 # Fallback default
        max_saved = 0

    print(f"Optimization Results: Peak Churn Reduction at Top {optimal_pct:.0%}")
    print(f"(Est. {max_saved:.1f} members saved in training set)")

    # Plot the Optimization Curve
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    # 1. The Line
    plt.plot(x_axis, net_impacts, color='#2c3e50', linewidth=2.5, zorder=2)
    
    # 2. The "Value Area" (Green Zone up to Optimal)
    plt.fill_between(x_axis, 0, net_impacts, 
                     where=(np.array(x_axis) <= optimal_pct + 0.01), 
                     color='#27ae60', alpha=0.3, label='Net Value Created')
    
    # 3. The "Waste Area" (Grey Zone after Optimal)
    plt.fill_between(x_axis, 0, net_impacts, 
                     where=(np.array(x_axis) >= optimal_pct), 
                     color='#95a5a6', alpha=0.2, label='Diminishing Returns')
    
    # 4. The Peak Marker & Line
    plt.axvline(optimal_pct, color='#c0392b', linestyle='--', linewidth=2)
    plt.scatter([optimal_pct], [max_saved], color='#c0392b', s=100, zorder=3, edgecolors='white', linewidth=2)
    
    # 5. Annotations
    plt.text(optimal_pct, max_saved + (max(net_impacts)*0.05), 
             f"Optimal Cutoff: {optimal_pct:.0%}\n(Max Net Members Saved)", 
             ha='center', fontweight='bold', color='#c0392b')
    
    plt.text(optimal_pct/2, max(net_impacts)/2, 
             "Profitability Zone\n(Outreach > Churn Risk)", 
             ha='center', color='#1e8449', fontweight='bold')
    
    # Formatting
    plt.title("Optimization Model: Maximizing Total Net Members Saved", fontsize=14, fontweight='bold')
    plt.xlabel("% of Population Targeted", fontsize=12)
    plt.ylabel("Est. Net Members Saved (OOF)", fontsize=12)
    
    # Format axes as percentages
    
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.ylim(0, max(net_impacts)*1.2)
    plt.legend(loc='upper right')
    # plt.show()
    plt.savefig(f"{PROJECT_ROOT}/output/optimal_n.png", dpi=300)

    
    return optimal_pct


# --- STEP 4: FINAL RETRAINING ---
def retrain_final_models(X, y_ret, t, best_model_c, best_model_t):
    """
    Retrains the best estimators on the full dataset.
    """
    print("\n--- 4. Retraining Final Models on Full Data ---")
    best_model_c.fit(X[t == 0], y_ret[t == 0])
    best_model_t.fit(X[t == 1], y_ret[t == 1])
    
    return best_model_c, best_model_t