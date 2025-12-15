import pandas as pd

OBSERVATION_END_DATE = pd.Timestamp('2025-07-15').tz_localize('UTC')

def build_features(df_churn, df_app, df_web, df_claims, input_risk_titles=None):
    """
    Args:
        input_risk_titles (list): 
            - If None: Runs in 'TRAINING' mode. Calculates correlations with Churn 
              to identify risk titles. Returns (df, learned_titles).
            - If List: Runs in 'TEST' mode. Uses the provided list to flag risk. 
              Returns (df, input_risk_titles).
    """

    # --- 1. BASE TABLE ---
    # Handle case where 'churn' might not exist in production/test data
    cols_to_keep = ['member_id', 'outreach']
    if 'churn' in df_churn.columns:
        cols_to_keep.append('churn')
        
    df_churn['tenure_days'] = (OBSERVATION_END_DATE - df_churn['signup_date']).dt.days
    features = df_churn[cols_to_keep + ['tenure_days']].copy()

    # --- 2. APP USAGE ---
    app_agg = df_app.groupby('member_id').agg(
        total_sessions=('event_type', 'count'),
        last_session=('timestamp', 'max')
    ).reset_index()
    app_agg['days_since_last_session'] = (OBSERVATION_END_DATE - app_agg['last_session']).dt.days
    app_agg = app_agg.drop(columns=['last_session'])

    # --- 3. CLAIMS ---
    claims_agg = df_claims.groupby('member_id').agg(
        total_claims=('icd_code', 'count'),
        unique_diagnoses=('icd_code', 'nunique')
    ).reset_index()

    claims_last = df_claims.groupby('member_id')['diagnosis_date'].max().reset_index()
    claims_last['days_since_last_claim'] = (OBSERVATION_END_DATE - claims_last['diagnosis_date']).dt.days
    claims_last = claims_last.drop(columns=['diagnosis_date'])

    claims_final = claims_agg.merge(claims_last, on='member_id', how='left')

    # --- 4. WEB VISITS (LOGIC SWITCH) ---
    
    risk_titles = []
    
    # MODE A: TRAINING (We need to learn the titles)
    if input_risk_titles is None:
        print("Processing Robust Features [TRAIN MODE] - Calculating Correlations...")
        
        # 1. Pivot
        web_pivot = pd.crosstab(df_web['member_id'], df_web['title'])
        
        # 2. Merge with Churn labels
        # Note: We must have 'churn' in df_churn for this to work
        correlation_df = df_churn[['member_id', 'churn']].merge(
            web_pivot, on='member_id', how='left'
        ).fillna(0)
        
        # 3. Correlation
        corrs = correlation_df.drop(columns=['member_id']).corr()['churn'].drop('churn')
        
        # 4. Define Lists
        risk_titles = corrs[corrs > 0].index.tolist()
        health_titles = corrs[corrs <= 0].index.tolist() # Just for logging
        
        print(f"   -> Learned {len(risk_titles)} Risk Titles (Pos Corr)")
        
    # MODE B: TEST/PROD (Use existing knowledge)
    else:
        print("Processing Robust Features [TEST MODE] - Using Pre-defined Rules...")
        risk_titles = input_risk_titles
        print(f"   -> Applied {len(risk_titles)} Risk Titles from input")

    # --- APPLY WEB FEATURES ---
    # Determine flags based on the 'risk_titles' list (either learned or provided)
    df_web['is_risk'] = df_web['title'].isin(risk_titles).astype(int)
    # Note: We don't need a health list, anything not risk is health
    df_web['is_health'] = (~df_web['title'].isin(risk_titles)).astype(int)

    web_agg = df_web.groupby('member_id').agg(
        total_web_visits=('title', 'count'),
        total_risk_visits=('is_risk', 'sum'),
        total_health_visits=('is_health', 'sum')
    ).reset_index()

    web_last = df_web.groupby('member_id')['timestamp'].max().reset_index()
    web_last['days_since_last_web'] = (OBSERVATION_END_DATE - web_last['timestamp']).dt.days
    web_last = web_last.drop(columns=['timestamp'])

    web_final = web_agg.merge(web_last, on='member_id', how='left')

    # --- 5. MERGE ALL ---
    master_df = features.merge(app_agg, on='member_id', how='left')
    master_df = master_df.merge(claims_final, on='member_id', how='left')
    master_df = master_df.merge(web_final, on='member_id', how='left')

    # --- 6. CLEANUP ---
    count_cols = ['total_sessions', 'total_claims', 'unique_diagnoses', 
                  'total_web_visits', 'total_risk_visits', 'total_health_visits']
    existing_count_cols = [c for c in count_cols if c in master_df.columns]
    master_df[existing_count_cols] = master_df[existing_count_cols].fillna(0)

    recency_cols = [c for c in master_df.columns if 'days_since' in c]
    master_df[recency_cols] = master_df[recency_cols].fillna(999)

    master_df['claims_per_day'] = master_df['total_claims'] / (master_df['tenure_days'] + 1)
    master_df['sessions_per_day'] = master_df['total_sessions'] / (master_df['tenure_days'] + 1)
    master_df['risk_ratio'] = master_df['total_risk_visits'] / (master_df['total_web_visits'] + 1)

    # Remove non-robust features
    to_drop = [
        'tenure_days',
        'total_web_visits',
        'total_risk_visits',
        'total_health_visits',
        'days_since_last_web',
        'days_since_last_claim'
    ]
    master_df = master_df.drop(columns=to_drop, errors='ignore')

    # IMPORTANT: Return BOTH the dataframe AND the learned list
    return master_df, risk_titles