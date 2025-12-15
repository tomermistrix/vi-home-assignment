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
3.  Generate a figure plot `optimal_n.png` in the `output` directory, to determine optimal n. 
4.  Generate a file named `outreach_list.csv` in the `output` directory, containing the prioritized list of members for the outreach campaign. The list includes `member_id`, `prioritization_score`, and `rank`. 

My produced results are available at `output_reference`.

## Approach & Methodology
### Feature Engineering
Our feature strategy prioritized **behavioral intensity** and **current user intent** over historical accumulation. Since Uplift models can easily overfit to strong "Base Risk" signals (like Tenure), we rigorously curated the feature set to focus on responsiveness.
1. **Addressing Data Quality: Aggregation over Sparsity**

    Raw transactional data (e.g., specific ICD-10 codes or 25+ unique web titles) proved too sparse for the dataset size, leading to high variance in the test set.
    - **Action:** We replaced specific counts with robust aggregates. For example, rather than tracking 10 different diagnosis codes, we calculated `total_claims` and `unique_diagnoses`.
2. **Domain Relevance: Intent Clustering**

   We analyzed the semantic meaning of web visits to distinguish between "Engagement" and "Disengagement."
   - **Health-Related Titles** (e.g., Diabetes Management) correlated with retention.
   - **Leisure/Risk Titles** (e.g., Game Reviews, Match Highlights) correlated with churn.
   - **Action:** We aggregated these into `risk_ratio` features, capturing the proportion of a user's activity that is high-risk, rather than just the raw volume.
3. **Predictive Power: Prioritizing Intensity (Ratios vs. Totals)**

   During model evaluation, we observed that raw cumulative features (e.g., `tenure_days`, `total_web_visits`) biased the model towards "Loyalty" rather than "Persuadability." A user with 100 visits over 5 years is very different from a user with 100 visits in 1 month.
    - **Action:** We removed raw cumulative totals (including `tenure_days` and `total_visits`) and replaced them with Intensity Ratios (e.g., `claims_per_day`, `sessions_per_day`).
    - **Why:** This forces the T-Learner to focus on the user's current state of need and interaction velocity, which are stronger predictors of whether they will respond to an immediate outreach intervention.

### Causal Inference Modeling (Uplift)
We formulated this as a Causal Inference problem to estimate the Conditional Average Treatment Effect (CATE) of outreach.

**Handling Data Imbalance**: The dataset is highly unbalanced, with approximately 80% of members not churning and 20% churning. To prevent the model from developing a "lazy" solution (i.e., classifying all members as non-churners), we scaled the sample weights during model training. This ensures that the minority class (churners) is given appropriate importance.

**Why not S-Learner (Single Model)?**

Initial experimentation with an S-Learner (where outreach is a feature) yielded poor ranking performance. Because "Base Churn Risk" (e.g., medical history) is a much stronger signal than the "Outreach" signal, the S-Learner treated the intervention as noise.

**Solution: T-Learner (Two-Model Approach)**

We employed a T-Learner architecture, training two separate XGBoost classifiers:
- **Model Control**: Learns natural churn behavior for ignored members.
- **Model Treated**: Learns churn behavior for contacted members.
- **Uplift Score**: $P(\text{Stay} | \text{Treated}) - P(\text{Stay} | \text{Control})$.
This forced the system to explicitly model the difference in outcomes, successfully identifying "Persuadables" (High Uplift) vs. "Sleeping Dogs" (Negative Uplift). Members are ranked for prioritized outreach based on the uplift score.

### Optimal number of users (Determining n)

We determined the optimal outreach size not just by cost, but also by **Customer Sentiment Risk**.
- **Beyond Cost:** Since outreach costs are marginal, a purely cost-driven model would suggest targeting nearly everyone. However, this ignores the "Sleeping Dog" effect-users who churn because they were contacted.
- **The Strategy:** We optimized for **Total Net Churn Reduction**. By plotting the cumulative "Members Saved" curve on Out-of-Fold (OOF) predictions, we identified the inflection point (65%) where the strategy shifts from value-generating to value-destroying.
- **Result:** This cutoff maximizes the volume of saved customers while explicitly filtering out the bottom 35%, where outreach was proven to have a negative impact.

## Analysis of Outreach List and Optimal n:
### Top 15 Users to Contact (Highest Uplift)

These are the users with the highest predicted uplift scores - the ones most likely to benefit from outreach.

| User ID | Outreach | Prob. Churn if Ignored | Prob. Churn if Treated | Uplift Score | Actual Churn |
|---------|----------|-----------------------|-----------------------|--------------|--------------|
| 9861    | 0        | 0.7096                | 0.4343                | 0.2753       | 0            |
| 7004    | 1        | 0.6117                | 0.3454                | 0.2664       | 0            |
| 572     | 0        | 0.6734                | 0.4178                | 0.2556       | 0            |
| 8755    | 0        | 0.6531                | 0.4033                | 0.2498       | 0            |
| 846     | 1        | 0.6246                | 0.3749                | 0.2497       | 0            |
| 4758    | 1        | 0.6239                | 0.3749                | 0.2490       | 0            |
| 4385    | 0        | 0.6105                | 0.3627                | 0.2478       | 0            |
| 1892    | 0        | 0.6889                | 0.4412                | 0.2477       | 0            |
| 6685    | 0        | 0.6607                | 0.4204                | 0.2403       | 0            |
| 2383    | 0        | 0.6503                | 0.4119                | 0.2384       | 0            |
| 799     | 0        | 0.6771                | 0.4397                | 0.2374       | 1            |
| 1487    | 1        | 0.6384                | 0.4011                | 0.2373       | 0            |
| 7179    | 0        | 0.6384                | 0.4037                | 0.2347       | 0            |
| 1752    | 0        | 0.6503                | 0.4164                | 0.2340       | 1            |
| 8577    | 0        | 0.6607                | 0.4273                | 0.2335       | 0            |

**Observation:**  

Among the top 15 users to contact, the **only users who actually churned (user IDs 799 and 1752) were not outreached**. This aligns with expectations: the model correctly identifies high-risk users who benefit from outreach.

---

### Top 15 Sleeping Dogs (Negative Uplift)

These are users for whom outreach may **increase churn risk**.

| User ID | Outreach | Prob. Churn if Ignored | Prob. Churn if Treated | Uplift Score | Actual Churn |
|---------|----------|-----------------------|-----------------------|--------------|--------------|
| 6623    | 1        | 0.4082                | 0.5540                | -0.1458      | 1            |
| 8619    | 1        | 0.3119                | 0.4578                | -0.1460      | 0            |
| 5454    | 0        | 0.3378                | 0.4857                | -0.1479      | 0            |
| 2808    | 1        | 0.3253                | 0.4761                | -0.1507      | 0            |
| 5259    | 1        | 0.4000                | 0.5540                | -0.1541      | 0            |
| 8831    | 0        | 0.3402                | 0.4960                | -0.1557      | 0            |
| 5795    | 0        | 0.3770                | 0.5332                | -0.1562      | 0            |
| 8702    | 0        | 0.3280                | 0.4867                | -0.1588      | 0            |
| 1045    | 0        | 0.3860                | 0.5459                | -0.1599      | 0            |
| 8461    | 1        | 0.3545                | 0.5160                | -0.1615      | 1            |
| 4773    | 1        | 0.3226                | 0.4857                | -0.1631      | 1            |
| 8088    | 0        | 0.3437                | 0.5076                | -0.1640      | 0            |
| 8560    | 1        | 0.3882                | 0.5531                | -0.1649      | 0            |
| 6815    | 1        | 0.3676                | 0.5332                | -0.1656      | 0            |
| 1023    | 0        | 0.3150                | 0.4809                | -0.1659      | 0            |

**Observation:**

Among the sleeping dogs, the **users who actually churned (6623, 8461, 4773) were all outreached**, suggesting that outreach to these users could have **adversely impacted churn**, which is consistent with the negative uplift scores.

---

### Top K Uplift Evaluation

| Top % | n Users | Actual Lift | Control Churn | Treated Churn |
|-------|---------|------------|---------------|---------------|
| 10%   | 1,000   | +2.73%     | 20.1%         | 17.4%         |
| 20%   | 2,000   | +5.76%     | 20.7%         | 14.9%         |
| 30%   | 3,000   | +3.96%     | 20.9%         | 16.9%         |
| 40%   | 4,000   | +2.70%     | 21.2%         | 18.5%         |
| 45%   | 4,500   | +1.88%     | 20.8%         | 18.9%         |
| 50%   | 5,000   | +1.81%     | 20.9%         | 19.1%         |
| 55%   | 5,500   | +1.82%     | 21.1%         | 19.3%         |
| 60%   | 6,000   | +1.39%     | 20.9%         | 19.5%         |
| 65%   | 6,500   | +1.53%     | 20.8%         | 19.2%         |
| 70%   | 7,000   | +1.54%     | 20.9%         | 19.3%         |
| 75%   | 7,500   | +0.94%     | 20.6%         | 19.6%         |
| 80%   | 8,000   | +0.69%     | 20.3%         | 19.6%         |
| 90%   | 9,000   | +0.52%     | 20.2%         | 19.7%         |
| 100%  | 10,000  | +0.48%     | 20.2%         | 19.7%         |

**Observation:**  
- The **optimal top-K percentage is ~65%**.  
- After 65%, the **lift stabilizes**, indicating diminishing returns from outreaching additional users.  
- This supports using the **top 65% of users** for outreach as a practical and effective strategy.

---

### Summary Insights

1. **High Uplift Users:** Targeting these users is effective; the few actual churners were not contacted, showing the model identifies who truly benefits.  
2. **Sleeping Dogs:** Outreach to these users may be counterproductive, as evidenced by the churned users being outreached.  
3. **Top-K Analysis:** Selecting the top 65% of users maximizes lift while avoiding wasted outreach, after which additional outreach gives minimal incremental benefit.

### Test Evaluation and Comparison to Baseline
To evaluate the model and compare to the baseline using standard metrics as classification report and AUC, we derive the predicted churn labels as follows:
1. If the member was outreached, we predict the churn probability using **Model Treated**.
2. If the member was not outreached, we predict the churn probability using **Model Control**.

#### ROC-AUC Comparison
- **Test Set AUC (Our Model):** 0.6602  
- **Test Set AUC (Baseline):** 0.4891  
- **AUC Improvement:** **+0.1711**

---

#### Classification Report - Our Model

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| no_churn  | 0.86      | 0.63   | 0.73     | 7,996   |
| churn     | 0.29      | 0.61   | 0.39     | 2,004   |
| **Accuracy** |           |        | **0.62** | 10,000  |
| **Macro Avg** | 0.58      | 0.62   | 0.56     | 10,000  |
| **Weighted Avg** | 0.75      | 0.62   | 0.66     | 10,000  |

---

#### Classification Report - Baseline

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| no_churn  | 0.80      | 0.51   | 0.62     | 7,996   |
| churn     | 0.20      | 0.48   | 0.28     | 2,004   |
| **Accuracy** |           |        | **0.50** | 10,000  |
| **Macro Avg** | 0.50      | 0.49   | 0.45     | 10,000  |
| **Weighted Avg** | 0.68      | 0.50   | 0.55     | 10,000  |

---

#### Summary
- Our model significantly outperforms the baseline in **ROC-AUC (+0.17)**.
- It improves **recall for the churn class** (0.61 vs. 0.48), capturing more churners.
- Overall accuracy increases from **50% → 62%**, indicating meaningful predictive gains.
