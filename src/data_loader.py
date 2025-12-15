import pandas as pd

def load_dataset(data_path, test=False):
    """Loads raw CSVs from a directory and parses dates."""
    print(f"Loading data from {data_path}...")

    if test:
        data_path = f"{data_path}test_"
    
    # Load tables
    df_churn = pd.read_csv(f"{data_path}churn_labels.csv")
    df_app = pd.read_csv(f"{data_path}app_usage.csv")
    df_web = pd.read_csv(f"{data_path}web_visits.csv")
    df_claims = pd.read_csv(f"{data_path}claims.csv")

    # use utc=True to ensure all datetimes are timezone-aware

    # 1. Churn Labels
    df_churn['signup_date'] = pd.to_datetime(df_churn['signup_date'], utc=True)
    # 2. App Usage
    df_app['timestamp'] = pd.to_datetime(df_app['timestamp'], utc=True)
    # 3. Web Visits
    df_web['timestamp'] = pd.to_datetime(df_web['timestamp'], utc=True)
    # 4. Claims
    # diagnosis_date is usually just a date (YYYY-MM-DD).
    # We convert to datetime, force UTC, then Normalize (sets time to 00:00:00)
    df_claims['diagnosis_date'] = pd.to_datetime(df_claims['diagnosis_date'], utc=True)

    print("Data loaded successfully (All dates set to UTC).")
    return df_churn, df_app, df_web, df_claims