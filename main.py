import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Waze Churn Analysis Dashboard",
    layout="wide"
)

# ---------- DATA LOADING ----------
DATA_PATH = "waze_churn_clean.csv"   # <-- change this to your actual file path

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)

# Assume target column is named 'churn' (0 = retained, 1 = churned)
# Drop obvious non-feature columns if present (e.g., 'user_id')
label_col = "churn"
drop_cols = [label_col]
for col in ["user_id", "ID", "id"]:
    if col in df.columns:
        drop_cols.append(col)

X = df.drop(columns=drop_cols)
y = df[label_col]

# Train/test split for model tab
@st.cache_data
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Simple tuned XGBoost (based on your notebook tuning)
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1,
        max_depth=8,
        min_child_weight=1,
        learning_rate=0.1,
        n_estimators=400,
    )

    xgb_clf.fit(X_train, y_train)

    y_test_pred = xgb_clf.predict(X_test)
    y_test_proba = xgb_clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
    }

    return xgb_clf, X_train, X_test, y_train, y_test, y_test_pred, y_test_proba, metrics

xgb_clf, X_train, X_test, y_train, y_test, y_test_pred, y_test_proba, model_metrics = train_model(X, y)

# ---------- HEADER ----------
with st.container():
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_center:
        st.markdown(
            """
            <h1 style="text-align: center; margin-bottom: 0;">
                Waze Churn Analysis Dashboard
            </h1>
            <p style="text-align: center; font-size: 1.05rem; margin-top: 0.35rem;">
                Understanding which behaviors drive churn and how PMs can act on them.
            </p>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ---------- TOP-LEVEL KPIs ----------
churn_rate = df[label_col].mean()
total_users = len(df)
churned_users = df[label_col].sum()
retained_users = total_users - churned_users

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.metric("Total Users", f"{total_users:,}")
with kpi_col2:
    st.metric("Churn Rate", f"{churn_rate*100:.1f}%")
with kpi_col3:
    st.metric("Churned Users", f"{churned_users:,}")

st.markdown("---")

# ---------- TABS ----------
tab_overview, tab_behavior, tab_model = st.tabs(
    ["Overview", "Behavior Explorer", "Model Insights"]
)

# ===== TAB 1: OVERVIEW =====
with tab_overview:
    st.header("📌 Churn Overview")

    col1, col2 = st.columns([1.2, 2])

    with col1:
        st.subheader("Churn vs Retained")

        churn_counts = df[label_col].value_counts().rename(index={0: "Retained", 1: "Churned"})
        churn_df = churn_counts.reset_index()
        churn_df.columns = ["Status", "Count"]

        fig_pie = px.pie(
            churn_df,
            names="Status",
            values="Count",
            hole=0.4,
            title="User Status Breakdown",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Key Takeaways (for PMs)")
        st.markdown(
            """
            - About **{:.1f}%** of users in this sample are labeled as churned.  
            - Churn is not evenly distributed: it is tied to **usage behavior**, not device type.  
            - This dashboard helps answer: *“Which behaviors signal churn risk, and how can we intervene?”*
            """.format(churn_rate * 100)
        )

# ===== TAB 2: BEHAVIOR EXPLORER =====
with tab_behavior:
    st.header("👣 Behavior Explorer")

    st.markdown(
        """
        Compare the distribution of key behavioral features between **retained** and **churned** users.  
        This is where PMs can see how habits differ across segments.
        """
    )

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    col_left, col_right = st.columns([1.2, 2])

    with col_left:
        feature = st.selectbox(
            "Select a feature to explore:",
            options=numeric_cols,
            index=0 if numeric_cols else None,
        )

        bins = st.slider(
            "Number of bins",
            min_value=10,
            max_value=60,
            value=30,
        )

    with col_right:
        if feature:
            df_plot = df[[feature, label_col]].copy()
            df_plot["Status"] = df_plot[label_col].map({0: "Retained", 1: "Churned"})

            fig_hist = px.histogram(
                df_plot,
                x=feature,
                color="Status",
                nbins=bins,
                barmode="overlay",
                marginal="box",
                opacity=0.65,
                labels={feature: feature},
            )

            fig_hist.update_layout(
                title=f"Distribution of {feature} by Churn Status",
                xaxis_title=feature,
                yaxis_title="Count",
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown(
                """
                **How to read this:**  
                - Look for where the **churned** distribution differs from **retained**.  
                - Features where churned users cluster in different ranges are strong candidates for PM focus.
                """
            )
        else:
            st.info("No numeric features available to plot.")

# ===== TAB 3: MODEL INSIGHTS =====
with tab_model:
    st.header("🧠 Model Insights – Tuned XGBoost")

    st.markdown(
        """
        The model here is a **tuned XGBoost classifier**, trained on user behavior features.  
        It’s used to (1) understand which features matter most, and (2) provide a practical way to flag at-risk users.
        """
    )

    # --- Metrics row ---
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Accuracy (Test)", f"{model_metrics['accuracy']*100:.1f}%")
    with m_col2:
        st.metric("Precision (Test)", f"{model_metrics['precision']*100:.1f}%")
    with m_col3:
        st.metric("Recall (Test)", f"{model_metrics['recall']*100:.1f}%")
    with m_col4:
        st.metric("F1-score (Test)", f"{model_metrics['f1']*100:.1f}%")

    st.markdown("---")

    # --- Confusion Matrix & PR Curve ---
    cm_col, pr_col = st.columns([1.2, 1.8])

    with cm_col:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_test_pred)
        cm_labels = ["Retained", "Churned"]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=cm_labels,
                y=cm_labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
            )
        )
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown(
            """
            - **TP (bottom-right):** churners correctly flagged.  
            - **FP (top-right):** non-churners we would unnecessarily target.  
            - **FN (bottom-left):** churners we miss – most costly for retention.
            """
        )

    with pr_col:
        st.subheader("Precision–Recall Curve")

        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        avg_prec = average_precision_score(y_test, y_test_proba)

        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name="Precision–Recall curve",
            )
        )
        fig_pr.update_layout(
            title=f"Precision–Recall Curve (AP = {avg_prec:.3f})",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
        )
        st.plotly_chart(fig_pr, use_container_width=True)

        st.markdown(
            """
            This curve helps PMs decide **how aggressive** the model should be:
            - Moving right = higher recall (catch more churners)  
            - But precision drops (more false alarms)
            """
        )

    st.markdown("---")

    # --- Feature Importance ---
    st.subheader("Feature Importance – What Drives Churn?")

    feature_importances = xgb_clf.feature_importances_
    feature_names = X_train.columns

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances
    }).sort_values(by="importance", ascending=False)

    top_n = st.slider(
        "Number of top features to show:",
        min_value=5,
        max_value=min(20, len(fi_df)),
        value=10,
    )

    fi_top = fi_df.head(top_n).sort_values(by="importance", ascending=True)

    fig_fi = go.Figure()
    fig_fi.add_trace(
        go.Bar(
            x=fi_top["importance"],
            y=fi_top["feature"],
            orientation="h",
        )
    )
    fig_fi.update_layout(
        title=f"Top {top_n} Features Driving Churn (XGBoost)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
    )

    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        """
        **PM Interpretation:**  
        - Features at the top of this list are the **strongest behavioral signals** of churn.  
        - These are the levers Waze PMs should:
          - Monitor as early warning signals  
          - Target with habit-building nudges and experiments  
        """
    )
