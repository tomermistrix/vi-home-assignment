from src.data_loader import load_dataset
from src.evaluation import align_test_cols, evaluate_test_performance
from src.features import build_features
from src.model import get_trained_model
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = f"{PROJECT_ROOT}/output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def main():
    # 1. Load Data
    churn_tr, app_tr, web_tr, claims_tr = load_dataset(f"{PROJECT_ROOT}/data/train/")
    churn_te, app_te, web_te, claims_te = load_dataset(f"{PROJECT_ROOT}/data/test/", test=True)
    
    # 2. Feature Engineering
    df_train, risk_titles = build_features(churn_tr, app_tr, web_tr, claims_tr)
    df_test, _ = build_features(churn_te, app_te, web_te, claims_te, input_risk_titles=risk_titles)
    df_test = align_test_cols(df_train, df_test)
    
    # Prepare Arrays
    feature_cols = [c for c in df_train.columns if c not in ['member_id', 'churn', 'outreach', 'signup_date']]
    treatment_col = 'outreach'
    target_col = 'churn'
    
    # 3. Model Training & Optimization
    model, optimal_pct = get_trained_model(df_train, feature_cols, treatment_col, target_col, param_grid=None, random_state=42)
    
    print(f"Optimal Strategy: Target Top {optimal_pct:.1%}")

    # 4. Evaluate test set
    results = evaluate_test_performance(model, df_test, feature_cols, treatment_col, target_col)

    # 5. Generate List
    results_top_n = df_test[['member_id']].copy()
    results_top_n['prioritization_score'] = results['uplift_score']
    results_top_n = results_top_n.sort_values('prioritization_score', ascending=False)
    
    final_n = int(len(results_top_n) * optimal_pct)
    outreach = results_top_n.head(final_n).copy()
    outreach['rank'] = range(1, len(outreach)+1)
    
    outreach.to_csv(f"{PROJECT_ROOT}/output/outreach_list.csv", index=False)
    print("Success.")

if __name__ == "__main__":
    main()