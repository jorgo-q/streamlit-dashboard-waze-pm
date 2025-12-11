# 🚖 Waze  User Churn Prediction & Behavioral Insights Dashboard

---

### Streamlit Dashboard + Machine Learning Analysis

This project analyzes **user churn behavior for Waze**, using a public synthetic dataset inspired by Google Analytics training data.
The goal is not just to build a predictive model — but to understand **what behaviors actually drive churn**, and how Product Managers can act on them.

A **Streamlit dashboard** is included to let PMs, analysts, and stakeholders explore user behavior, model results, and churn-driving features interactively.

---

## 🚀 Live Dashboard (Streamlit)

👉 **[View the Streamlit Dashboard](https://waze-pm.streamlit.app/)**

---

# 📌 Project Overview

Waze wants to better understand:

* **Which user behaviors signal early churn risk?**
* **What features matter most for predicting churn?**
* **How can PMs use these insights to design interventions?**

This project walks through a full DS workflow:

1. Data loading & cleanup
2. Exploratory behavior analysis
3. Model development (Decision Tree, Random Forest, XGBoost)
4. Hyperparameter tuning
5. Feature importance interpretation
6. Product recommendations for early-warning churn detection

---

# 📊 Dataset

* ~195,000 users
* ~13 behavioral features (sessions, activity days, distance, engagement)
* Churn label (`0 = retained`, `1 = churned`)
* No demographic or marketing features — **behavior-only analysis**

These constraints make the problem realistic for analyzing **habit-driven churn**.

---

# 🧠 Modeling Approach

Three models were evaluated:

| Model             | Strengths                                          | Weaknesses                         |
| ----------------- | -------------------------------------------------- | ---------------------------------- |
| **Decision Tree** | Simple, interpretable                              | Overfits, unstable feature ranking |
| **Random Forest** | Strong accuracy, stable                            | Very low recall on churners        |
| **XGBoost**       | Best balance of metrics, robust feature importance | Needs tuning                       |

📌 **Tuned XGBoost was selected as the final model**
because it provides **reliable, consistent feature importance** — ideal for PM decision-making.

---

# 📈 Key Results

### Confusion Matrix (Tuned XGBoost)

```
Actual →   0      1
Pred  ↓
0        2785   524
1         156   110
```

**Interpretation:**

* ✔ Excellent at identifying retained users
* ✖ Misses many churners (524 false negatives)
* ✔ Feature importance still provides strong behavioral insights

### Main Metrics (Test Set)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 80.8% |
| Precision | 41.3% |
| Recall    | 17.3% |
| F1 Score  | 23.6% |

*Accuracy is high but misleading because churn is rare.*

---

# 🔍 Top Features Driving Churn (from XGBoost)

1. **Recent Sessions**
2. **Active Days Count**
3. **Session Trends / Consistency**
4. **Total Sessions**
5. **Engagement Indicators**

### PM Interpretation

Users don’t churn because of device type or trip length —
they churn when **habits fade**.

> Churn is fundamentally a **consistency problem**.

---

# 📊 Streamlit Dashboard

The dashboard is organized into three tabs:

---

## 1️⃣ Overview

* Churn rate
* Retained vs churned distribution
* Key high-level takeaways

---

## 2️⃣ Behavior Explorer

Compare retained vs churned users across any behavioral metric:

* Histograms
* Box plots
* Weekly/monthly patterns

Perfect for identifying **behavioral signals**.

---

## 3️⃣ Model Insights

* XGBoost performance metrics
* Confusion matrix
* Precision–Recall curve
* Feature importance (top behavioral drivers)

This helps PMs decide:

* *Who should be flagged for intervention?*
* *Which habits should we reinforce?*
* *When should the retention team take action?*

---

# 🎯 Product Recommendations

Based on feature importance and behavioral patterns:

### 1. **Early-Warning Triggers**

Flag users with declining recency or activity days.

### 2. **Habit Reinforcement**

Encourage users to save locations, plan routes, or use the app consistently.

### 3. **Targeted Interventions**

Push notifications, emails, or in-app nudges for at-risk users.

### 4. **A/B Testing**

Evaluate interventions on high-risk cohorts to measure retention lift.

---

# 🧱 Project Structure

```
📁 waze-churn-dashboard
│
├── dashboard/
│   └── main.py                 # Streamlit app
│
├── data/
│   └── waze_churn_clean.csv    # Dataset
│
├── notebooks/
│   └── Waze_Churn_Analysis.ipynb
│
├── README.md                   # Project overview
└── requirements.txt            # Packages for Streamlit / modeling
```

---

# 🛠️ Installation

```bash
git clone https://github.com/<your-username>/waze-churn-dashboard
cd waze-churn-dashboard

pip install -r requirements.txt

streamlit run dashboard/main.py
```

---

# 📚 Technologies Used

* **Python**
* **Pandas, NumPy**
* **XGBoost, Scikit-Learn**
* **Plotly, Seaborn**
* **Streamlit**
* **Jupyter Notebooks**

---

# 🏁 Conclusion

This project demonstrates how behavior-driven machine learning models can help Waze:

* Identify churn risk earlier
* Understand which habits matter most
* Build actionable PM strategies
* Drive long-term user engagement

While the predictive recall is still low, the **behavioral insights are strong**, and provide a solid foundation for improving retention systems at scale.
