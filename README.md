# WellCo Churn Prediction

## Project Overview

This project aims to help WellCo, an employer-sponsored healthcare ecosystem, reduce member churn. The primary objective is to identify members who are most likely to churn and would be most receptive to an outreach campaign. To achieve this, we build an uplift model to predict the impact of an outreach on a member's likelihood to churn. The model provides a ranked list of members for prioritized outreach and determines the optimal number of members to target.

The approach is as follows:
1.  **Data Loading and Preprocessing**: Load the provided datasets (`app_usage`, `web_visits`, `claims`, and `churn_labels`).
2.  **Feature Engineering**: Create features that capture member engagement, health profile, and other relevant information from the raw data.
3.  **Uplift Modeling**: Train a model to predict the uplift in churn probability due to an outreach. This helps in identifying members who are "persuadable".
4.  **Evaluation**: Evaluate the model's performance on a hold-out test set.
5.  **Prioritization**: Generate a ranked list of members to target for outreach based on their predicted uplift score.
6.  **Optimal Outreach Size**: Determine the optimal number of members to contact to maximize the return on investment of the outreach campaign.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tomermistrix/vi-home-assignment.git
    cd vi-home-assignment
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Run Instructions

To run the entire pipeline (from data loading to generating the outreach list), execute the `main.py` script:

```bash
python main.py
```

This will:
1.  Load and process the training and test data from the `data/` directory.
2.  Train the uplift model.
3.  Generate a file named `outreach_list.csv` in the root directory, containing the prioritized list of members for the outreach campaign. The list includes `member_id`, `prioritization_score`, and `rank`.


## Approach & Methodology
### Feature Engineering
Our feature strategy focused on behavioral intensity and user intent rather than raw transactional volume.
1. **Aggregations over Sparsity**: We initially explored using specific ICD codes and Web Titles as features. However, analysis showed these were too sparse for the test set size, leading to overfitting. We replaced specific counts with robust aggregates (e.g., total_claims, total_sessions) and recency metrics (e.g., days_since_last_session).
2. **Intent Clustering (Web Visits)**: We discovered a strong semantic split in web activity.
Health-Related Titles (e.g., "Diabetes Management") had negative correlations with churn (Engagement signal).
Leisure/Risk Titles (e.g., "Game Reviews", "Match Highlights") had positive correlations with churn (Disengagement/Risk signal). Instead of using ~25 individual titles, we engineered two high-signal features: total_health_visits and total_risk_visits. This captured user intent while reducing dimensionality.
3. **Intensity Ratios**: We normalized counts by tenure (e.g., claims_per_day) to distinguish between long-term low-usage users and new high-intensity users.

### Causal Inference Modeling (Uplift)
We formulated this as a Causal Inference problem to estimate the Conditional Average Treatment Effect (CATE) of outreach.

**Why not S-Learner (Single Model)?**
Initial experimentation with an S-Learner (where outreach is a feature) yielded poor ranking performance. Because "Base Churn Risk" (e.g., medical history) is a much stronger signal than the "Outreach" signal, the S-Learner treated the intervention as noise, resulting in a flat Qini curve.

**Solution**: T-Learner (Two-Model Approach)

We employed a T-Learner architecture, training two separate XGBoost classifiers:
**Model Control**: Learns natural churn behavior for ignored members.
**Model Treated**: Learns churn behavior for contacted members.
**Uplift Score**: $P(\text{Stay} | \text{Treated}) - P(\text{Stay} | \text{Control})$.
This forced the system to explicitly model the difference in outcomes, successfully identifying "Persuadables" (High Uplift) vs. "Sleeping Dogs" (Negative Uplift).

**Optimization (Determining 'n')**: Since the outreach cost is marginal, we avoided hard ROI calculations and optimized for Algorithmic Efficiency.

We performed Out-of-Fold (OOF) Analysis on the training set to simulate unseen data.
We plotted the Net Churn Reduction Curve and selected the cutoff point where the marginal gain of targeting additional users diminished significantly compared to random selection.
This validated a strategy of targeting the Top 60% of the population, maximizing total churn reduction without wasting resources on users with zero or negative treatment effects.