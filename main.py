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

# ---------- CONSISTENT COLOR MAP (IMPORTANT) ----------
# Hard lock the colors so churned/retained never swap in plots.
STATUS_ORDER = ["Retained", "Churned"]
STATUS_COLOR_MAP = {
    "Retained": "#1f77b4",  # blue
    "Churned": "#ff7f0e",   # orange
}


# ---------- DATA LOADING ----------
DATA_PATH = "files/waze_dataset.csv"  # <-- your path

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only known labels
    if "label" in df.columns:
        df = df[df["label"].isin(["retained", "churned"])].copy()

    # Target: churned_1
    if "label" in df.columns and "churned_1" not in df.columns:
        df["churned_1"] = (df["label"] == "churned").astype(int)

    # Device encoding
    if "device" in df.columns and "device_new" not in df.columns:
        df["device_new"] = (df["device"] == "Android").astype(int)

    # Engineered features (safe division)
    if "percent_sessions" not in df.columns and {"sessions", "total_sessions"}.issubset(df.columns):
        df["percent_sessions"] = df["sessions"] / df["total_sessions"].replace(0, np.nan)
        df["percent_sessions"] = df["percent_sessions"].fillna(0)

    if "total_sessions_per_day" not in df.columns and {"total_sessions", "n_days_after_onboarding"}.issubset(df.columns):
        df["total_sessions_per_day"] = df["total_sessions"] / df["n_days_after_onboarding"].replace(0, np.nan)
        df["total_sessions_per_day"] = df["total_sessions_per_day"].fillna(0)

    if "kms_driving_day" not in df.columns and {"driven_km_drives", "driving_days"}.issubset(df.columns):
        df["kms_driving_day"] = df["driven_km_drives"] / df["driving_days"].replace(0, np.nan)
        df["kms_driving_day"] = df["kms_driving_day"].fillna(0)

    # Drop obvious NaNs (optional safety)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0)

    return df


df = load_data(DATA_PATH)

# ---------- FEATURES / TARGET ----------
label_col = "churned_1"

if label_col not in df.columns:
    st.error(f"Target column '{label_col}' not found in data. Check your CSV and path.")
    st.write("Available columns:", list(df.columns))
    st.stop()

drop_cols = [label_col]
for col in ["label", "device"]:
    if col in df.columns:
        drop_cols.append(col)
for col in ["user_id", "ID", "id"]:
    if col in df.columns and col not in drop_cols:
        drop_cols.append(col)

X = df.drop(columns=drop_cols)
y = df[label_col]


# ---------- MODEL TRAINING (WEIGHTED XGBOOST) ----------
@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )

    # scale_pos_weight = (# negatives) / (# positives) on TRAIN ONLY
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = (neg / pos) if pos > 0 else 1.0

    # Weighted XGBoost (simple + aligned with your notebook)
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        scale_pos_weight=spw
    )

    xgb_clf.fit(X_train, y_train)

    y_test_pred = xgb_clf.predict(X_test)
    y_test_proba = xgb_clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1": f1_score(y_test, y_test_pred, zero_division=0),
        "scale_pos_weight": spw
    }

    return xgb_clf, X_train, X_test, y_train, y_test, y_test_pred, y_test_proba, metrics


xgb_clf, X_train, X_test, y_train, y_test, y_test_pred, y_test_proba, model_metrics = train_model(X, y)


# ---------- HEADER ----------
with st.container():
    col_center = st.columns([1])[0]
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
churned_users = int(df[label_col].sum())
retained_users = total_users - churned_users

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.metric("Total Users", f"{total_users:,}")
with kpi_col2:
    st.metric("Churn Rate", f"{churn_rate * 100:.1f}%")
with kpi_col3:
    st.metric("Churned Users", f"{churned_users:,}")
with kpi_col4:
    st.metric("scale_pos_weight", f"{model_metrics['scale_pos_weight']:.2f}")

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
        churn_df = churn_counts.reindex(STATUS_ORDER).reset_index()
        churn_df.columns = ["Status", "Count"]

        fig_pie = px.pie(
            churn_df,
            names="Status",
            values="Count",
            hole=0.4,
            title="User Status Breakdown",
            color="Status",
            color_discrete_map=STATUS_COLOR_MAP
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Key Takeaways (for PMs)")
        st.markdown(
            f"""
            - About **{churn_rate * 100:.1f}%** of users in this sample are labeled as churned.  
            - The goal is **not** to claim a perfect churn predictor. The real value is explaining **which behaviors drive churn risk**.  
            - This dashboard focuses on: *“Which behaviors signal churn risk, and how can we intervene?”*
            """
        )


# ===== TAB 2: BEHAVIOR EXPLORER =====
with tab_behavior:
    st.header("👣 Behavior Explorer")

    st.markdown(
        """
        Compare the distribution of key behavioral features between **retained** and **churned** users.  
        This helps PMs see how habits differ across segments.
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
            df_plot["Status"] = pd.Categorical(df_plot["Status"], categories=STATUS_ORDER, ordered=True)

            fig_hist = px.histogram(
                df_plot,
                x=feature,
                color="Status",
                category_orders={"Status": STATUS_ORDER},
                color_discrete_map=STATUS_COLOR_MAP,
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
                legend_title_text="Status"
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown(
                """
                **How to read this:**  
                - Look for where the **churned** distribution shifts away from **retained**.  
                - Features where churned users cluster in different ranges are strong candidates for PM focus.
                """
            )
        else:
            st.info("No numeric features available to plot.")


# ===== TAB 3: MODEL INSIGHTS =====
with tab_model:
    st.header("🧠 Model Insights – XGBoost (scale_pos_weight)")

    st.markdown(
        """
        The model here is a **weighted XGBoost classifier** trained on user behavior features.  
        It is not a perfect churn predictor, but it is useful for:
        - spotting behavioral patterns linked to churn risk  
        - flagging at-risk users early (with an understandable trade-off in false positives)
        """
    )

    # --- Metrics row ---
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Accuracy (Test)", f"{model_metrics['accuracy'] * 100:.1f}%")
    with m_col2:
        st.metric("Precision (Test)", f"{model_metrics['precision'] * 100:.1f}%")
    with m_col3:
        st.metric("Recall (Test)", f"{model_metrics['recall'] * 100:.1f}%")
    with m_col4:
        st.metric("F1-score (Test)", f"{model_metrics['f1'] * 100:.1f}%")

    st.markdown("---")

    # --- Confusion Matrix & PR Curve ---
    cm_col, pr_col = st.columns([1.2, 1.8])

    with cm_col:
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_test_pred)
        cm_labels = ["Retained", "Churned"]  # 0 then 1

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
            - **TP (Churned → Churned):** churners correctly flagged  
            - **FP (Retained → Churned):** false alarms (we target users who wouldn’t churn)  
            - **FN (Churned → Retained):** missed churners (lost opportunity to intervene)  
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
                name="PR curve",
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
            This curve helps PMs decide how aggressive the model should be:
            - Moving right = higher recall (catch more churners)  
            - Precision drops (more false alarms)  
            """
        )

    st.markdown("---")

    # --- Feature Importance ---
    st.subheader("Feature Importance – What Drives Churn?")

    fi_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": xgb_clf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    top_n = st.slider(
        "Number of top features to show:",
        min_value=5,
        max_value=min(20, len(fi_df)),
        value=11,
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
        title=f"Top {top_n} Features Driving Churn (XGBoost - weighted)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=450,
    )

    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        """
        **PM Interpretation:**  
        - Top features are the strongest behavioral signals linked to churn risk.  
        - These are the levers Waze PMs should monitor and test interventions around  
          (habit-building, early onboarding success, and repeated navigation behaviors).  
        """
    )
