from src.optimization import tune_hyperparameters, generate_oof_scores, determine_optimal_threshold, retrain_final_models

def get_trained_model(df, feature_cols, treatment_col, target_col, param_grid=None, random_state=42):
    """
    End-to-End Uplift Pipeline (Refactored):
    1. Tunes Hyperparameters via CV.
    2. Generates OOF Scores to determine optimal 'n'.
    3. Determines optimal cutoff.
    4. Retrains final model on full data.
    """
    print("=== Starting End-to-End Uplift Pipeline (T-Learner) ===")

    # 0. Prepare Data Arrays
    X = df[feature_cols].values
    t = df[treatment_col].values
    y_churn = df[target_col].values
    # Target for training is Retention (1 = Stay)
    y_ret = 1 - y_churn

    # Default Params
    if param_grid is None:
        param_grid = {
          'max_depth': [2, 3, 4],
          'learning_rate': [0.01, 0.05, 0.1],
          'n_estimators': [50, 100, 150],
          'reg_lambda': [1, 5, 10]
      }

    # Step 1: Tune
    best_c, best_t = tune_hyperparameters(X, y_ret, t, param_grid, random_state)

    # Step 2: OOF Scores
    oof_scores = generate_oof_scores(X, y_ret, t, best_c, best_t, random_state)

    # Step 3: Optimization
    optimal_pct = determine_optimal_threshold(oof_scores, y_churn, t)

    # Step 4: Final Training
    final_model_tuple = retrain_final_models(X, y_ret, t, best_c, best_t)

    return final_model_tuple, optimal_pct