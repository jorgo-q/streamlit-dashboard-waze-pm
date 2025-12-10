import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------- Page config ----------
st.set_page_config(
    page_title="HBR - Uber Case Study Analysis Dashboard",
    layout="wide"
)

# ---------- DATA LOADING ----------
data_url = (
    "https://docs.google.com/spreadsheets/"
    "d/1fTpJACr1Ay6DEIgFxjFZF8LgEPiwwAFY/edit?usp=sharing&"
    "ouid=103457517634340619188&rtpof=true&sd=true"
)

@st.cache_data
def load_file(url: str):
    # Turn the "edit" URL into a direct xlsx export URL
    modified_url = url.split("/edit")[0] + "/export?format=xlsx"
    all_sheets = pd.read_excel(modified_url, sheet_name=None)
    return all_sheets

data = load_file(data_url)

# Convenience references
switchbacks_df = data["Switchbacks"].copy()
dict_sheet = data["Data Dictionary"]
copyright_sheet = data["Copyright"]

# ---------- HEADER ----------
with st.container():
    col_left, col_center, col_right = st.columns([1, 3, 1])

    with col_left:
        st.image("files/Uber-logo.png", use_container_width=True)

    with col_center:
        st.markdown(
            """
            <h1 style="text-align: center; margin-bottom: 0;">
                HBR - UBER Case Study Analysis Dashboard
            </h1>
            <p style="text-align: center; font-size: 1.05rem; margin-top: 0.35rem;">
                A simple dashboard built from a multi-sheet Excel file.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.image("files/rice-logo.jpg", use_container_width=True)

st.markdown("---")

# ---------- TABS ----------
tab_metadata, tab_dict, tab_viz = st.tabs(
    ["Metadata", "Data Dictionary", "Data Visualizations"]
)

# ===== TAB 1: METADATA (show_metadata) =====
with tab_metadata:
    st.header("📄 Metadata")

    # Replicates your show_metadata(data) logic, but in Streamlit
    # lines = [sentence[0] for sentence in data['Copyright'].dropna().values.tolist()]
    # Safer version:
    first_col_name = copyright_sheet.columns[0]
    lines = (
        copyright_sheet[first_col_name]
        .dropna()
        .astype(str)
        .tolist()
    )
    text = "\n\n".join(lines)

    st.markdown(text)

# ===== TAB 2: DATA DICTIONARY (show_dictionary) =====
with tab_dict:
    st.header("📚 Data Dictionary")

    # Replicates your show_dictionary(data) logic
    headers = dict_sheet.iloc[1, :].values.tolist()
    df_dict = dict_sheet.iloc[2:, :].copy()
    df_dict.columns = headers
    df_dict.reset_index(drop=True, inplace=True)

    # Optionally drop unnamed columns that come from blank Excel columns
    df_dict = df_dict.loc[:, ~df_dict.columns.str.contains("^Unnamed")]

    st.dataframe(df_dict, use_container_width=True)

# ===== TAB 3: DATA VISUALIZATIONS (Load data + Time series + Pie) =====
with tab_viz:
    st.header("📊 Data and Time Series")

    # We’ll work with a copy and ensure datetime is parsed once
    base_df = switchbacks_df.copy()
    base_df["period_start"] = pd.to_datetime(base_df["period_start"])

    col_data, col_ts, col_pie = st.columns([1.4, 2, 1.3])

    # ---------- Data + slider ----------
    with col_data:
        st.subheader("💿 Data")

        max_index = len(base_df) - 1
        start, end = st.slider(
            "Select row range to display",
            min_value=0,
            max_value=max_index,
            value=(0, min(50, max_index)),
        )

        # This filtered_df is used by ALL widgets / charts
        filtered_df = base_df.iloc[start : end + 1].copy()

        st.dataframe(filtered_df, use_container_width=True)

    # ---------- Time Series with metric selector ----------
    with col_ts:
        st.subheader("📈 Time Series of Uber Trips in Boston")

        candidate_metrics = [
            "trips_pool",
            "trips_express",
            "rider_cancellations",
        ]
        available_metrics = [
            c for c in candidate_metrics if c in filtered_df.columns
        ]

        selected_metrics = st.multiselect(
            "Select metrics to plot",
            options=available_metrics,
            default=available_metrics,  # show all three by default
        )

        fig_ts = go.Figure()

        for y_column in selected_metrics:
            fig_ts.add_trace(
                go.Scatter(
                    x=filtered_df["period_start"],
                    y=filtered_df[y_column],
                    mode="lines+markers",
                    name=y_column,
                )
            )

        fig_ts.update_layout(
            title="Time Series of Uber Metrics in Boston",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
        )

        if selected_metrics:
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("Select at least one metric to plot.")

    # ---------- Earnings Pie Chart ----------
    with col_pie:
        st.subheader("💰 Earnings Pie Chart")

        period = st.selectbox(
            "Select period:",
            options=["week", "month"],
            index=0,
        )

        df_pie = filtered_df.copy()

        if period == "week":
            df_pie["label"] = df_pie["period_start"].dt.day_name()
            pie_title = "Total Payouts per Weekday"
        else:
            df_pie["label"] = df_pie["period_start"].dt.month_name()
            pie_title = "Total Payouts per Month"

        if "total_driver_payout" in df_pie.columns:
            fig_pie = px.pie(
                df_pie,
                names="label",
                values="total_driver_payout",
                title=pie_title,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Column `total_driver_payout` not found in data.")
