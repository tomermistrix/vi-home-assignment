import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np


BASELINE_PATH = "data/baseline/"

def align_test_cols(train_df, test_df):
    """
    Ensures Test DF has exactly the same features as Train DF.
    """
    # Define metadata columns to exclude from 'features'
    # We handle member_id separately to ensure it's the first column
    metadata = ['member_id', 'churn', 'outreach', 'signup_date']

    # Extract only the feature columns from Train
    train_features = [c for c in train_df.columns if c not in metadata]

    # 1. Add missing columns to Test (fill with 0)
    for c in train_features:
        if c not in test_df.columns:
            test_df[c] = 0

    # 2. Build the Final Column List
    # Start with member_id
    cols_to_keep = ['member_id'] + train_features

    # Append targets if they exist in test (so we can evaluate later)
    if 'churn' in test_df.columns:
        cols_to_keep.append('churn')
    if 'outreach' in test_df.columns:
        cols_to_keep.append('outreach')

    # 3. Return the subset
    return test_df[cols_to_keep]


def evaluate_uplift_results(results_df, top_k=15):
    """
    Prints Top/Bottom users and Top-K Actual Lift analysis.
    """
    # Sort descending by uplift score
    sorted_df = results_df.sort_values(by='uplift_score', ascending=False)

    # 1. Inspect Persuadables (Top top_k)
    print(f"\n--- Top {top_k} Users to Contact (Highest Uplift) ---")
    cols_to_show = ['outreach', 'prob_churn_if_ignored', 'prob_churn_if_treated', 'uplift_score', 'actual_churn']
    print(sorted_df[cols_to_show].head(top_k))

    # 2. Inspect Sleeping Dogs (Bottom top_k)
    print(f"\n--- Top {top_k} Sleeping Dogs (Negative Uplift) ---")
    print(sorted_df[cols_to_show].tail(top_k))

    # 3. Top K Evaluation
    print("\n--- Top K Uplift Evaluation ---")
    total_rows = len(sorted_df)

    for k in [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]:
        n = int(total_rows * k)
        subset = sorted_df.head(n)

        # Split into Treated/Control within this Top K group
        subset_treated = subset[subset['outreach'] == 1]
        subset_control = subset[subset['outreach'] == 0]

        if len(subset_treated) > 0 and len(subset_control) > 0:
            c_treat = subset_treated['actual_churn'].mean()
            c_ctrl = subset_control['actual_churn'].mean()

            # Lift = Control - Treated (Positive is Good)
            lift = c_ctrl - c_treat

            print(f"Top {k*100:.0f}% (n={n}): Actual Lift = {lift:+.2%} (Control {c_ctrl:.1%} vs Treated {c_treat:.1%})")
        else:
            print(f"Top {k*100:.0f}%: Not enough data for comparison")

def evaluate_test_performance(model, df_test, feature_cols, treatment_col, target_col):
    """
    Evaluates the model on Test Data:
    1. Calculates Standard Metrics (AUC, Accuracy) against Baseline.
    2. Calculates Uplift Metrics (Top K, Persuadables) for Strategy.
    """
    print("\n--- Evaluating model against Baseline & Uplift Metrics ---")

    X_test = df_test[feature_cols].copy()
    y_true = df_test[target_col]
    actual_treatment = df_test[treatment_col].values

    # Initialize arrays
    prob_ret_control = np.zeros(len(df_test))
    prob_ret_treated = np.zeros(len(df_test))

    # --- STEP 1: GENERATE PROBABILITIES ---
    model_c, model_t = model
    prob_ret_control = model_c.predict_proba(X_test)[:, 1]
    prob_ret_treated = model_t.predict_proba(X_test)[:, 1]

    # Calculate Uplift (Benefit of Treating)
    # Benefit = P(Stay|Treat) - P(Stay|Control)
    uplift_scores = prob_ret_treated - prob_ret_control

    # --- STEP 2: STANDARD CHURN AUC (BASELINE COMPARISON) ---
    # We need the probability of CHURN given ACTUAL treatment
    # 1. Select the Retention Prob matching the actual treatment
    prob_ret_actual = np.where(actual_treatment == 1, prob_ret_treated, prob_ret_control)

    # 2. Convert to Churn Prob
    prob_churn_actual = 1 - prob_ret_actual

    auc = roc_auc_score(y_true, prob_churn_actual)
    print(f"Test Set AUC: {auc:.4f}")

    # BASELINE COMPARISON
    with open(f"{BASELINE_PATH}auc_baseline_test.txt", 'r') as f:
        baseline_auc_text = f.read()
        baseline_auc = float(baseline_auc_text.split('=')[-1])
    
    print(f"Baseline Test Set AUC: {baseline_auc:.4f}")
    print(f"AUC Difference (Our Model - Baseline): {auc - baseline_auc:+.4f}")


    print("\n--- Classification Report (Our Model) ---")
    print(classification_report(y_true, (prob_churn_actual > 0.5).astype(int), target_names=['no_churn', 'churn']))

    print("\n--- Classification Report (Baseline) ---")
    with open(f"{BASELINE_PATH}classification_report_baseline_test.txt", 'r') as f:
        print(f.read())


    # --- STEP 3: PREPARE DATAFRAME FOR VISUALIZER ---
    results_df = df_test.copy()
    results_df['uplift_score'] = uplift_scores
    results_df['actual_churn'] = y_true # Ensure column matches visualizer expectation

    # ADD MISSING COLUMNS (Converted to Churn Probs)
    results_df['prob_churn_if_ignored'] = 1 - prob_ret_control
    results_df['prob_churn_if_treated'] = 1 - prob_ret_treated

    # --- STEP 4: RUN UPLIFT VISUALIZER ---
    # This uses the function we defined earlier
    evaluate_uplift_results(results_df)

    return results_df