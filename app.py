from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="KRONOS - Gym Retention", layout="wide")

REQUIRED_COLUMNS = [
    "age",
    "months_since_joined",
    "join_month",
    "visits_per_week",
    "payment_delay_days",
]

MODEL_OPTIONS = {
    "Best Default": Path("cancellation_model.pkl"),
    "Logistic Regression": Path("cancellation_logistic_model.pkl"),
    "Random Forest": Path("cancellation_random_forest_model.pkl"),
    "Gradient Boosting": Path("cancellation_gradient_boosting_model.pkl"),
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def available_models():
    return {
        name: path
        for name, path in MODEL_OPTIONS.items()
        if path.exists()
    }


def predict_cancellation(model, df):
    return model.predict_proba(df[REQUIRED_COLUMNS])[:, 1] * 100


def get_risk(prob):
    if prob > 70:
        return "🔴 Critical Risk"
    if prob > 50:
        return "🟠 High Risk"
    if prob > 30:
        return "🟡 Medium Risk"
    return "🟢 Low Risk"


def get_season(month):
    if month in [1, 2]:
        return "❄️ New Year Resolutioner"
    if month in [6, 7, 8]:
        return "☀️ Summer Seeker"
    return "📅 Normal Joiner"


def add_prediction_columns(df, probabilities, model_label):
    result = df.copy()
    result["model_used"] = model_label
    result["cancellation_probability"] = np.round(probabilities, 1)
    result["risk_level"] = result["cancellation_probability"].apply(get_risk)
    result["join_season"] = result["join_month"].apply(get_season)
    return result


def member_display_columns(df):
    return [
        col
        for col in [
            "member_id",
            "age",
            "months_since_joined",
            "join_season",
            "visits_per_week",
            "payment_delay_days",
            "cancellation_probability",
            "risk_level",
            "model_used",
        ]
        if col in df.columns
    ]


def clean_label(value):
    text = str(value)
    for prefix in ["🔴 ", "🟠 ", "🟡 ", "🟢 ", "❄️ ", "☀️ ", "📅 "]:
        text = text.replace(prefix, "")
    return text


def to_download_csv(df):
    export_df = df.copy()
    for col in ["risk_level", "join_season"]:
        if col in export_df.columns:
            export_df[col] = export_df[col].apply(clean_label)
    return export_df.to_csv(index=False).encode("utf-8-sig")


def render_model_cards(model_configs, selected_model):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Models")
    for name in model_configs:
        if name == selected_model:
            st.sidebar.caption(f"{name} - Selected")
        else:
            st.sidebar.caption(name)


def render_dashboard(df):
    st.subheader("📊 Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Members", len(df))
    col2.metric("📈 Avg Cancellation Probability", f"{df['cancellation_probability'].mean():.1f}%")
    col3.metric("🔴 Critical Risk", len(df[df["risk_level"] == "🔴 Critical Risk"]))
    col4.metric("🟠 High Risk", len(df[df["risk_level"] == "🟠 High Risk"]))

    risk_order = ["🟢 Low Risk", "🟡 Medium Risk", "🟠 High Risk", "🔴 Critical Risk"]
    risk_summary = (
        df["risk_level"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .reset_index()
    )
    risk_summary.columns = ["Risk Level", "Count"]
    st.dataframe(risk_summary, use_container_width=True, hide_index=True)

    col_left, col_right = st.columns(2)
    with col_left:
        fig = px.histogram(
            df,
            x="cancellation_probability",
            nbins=20,
            title="Cancellation Probability Distribution",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.update_layout(
            xaxis_title="Cancellation Probability (%)",
            yaxis_title="Number of Members",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        season_stats = (
            df.groupby("join_season")["cancellation_probability"]
            .mean()
            .round(1)
            .reset_index()
        )
        fig = px.bar(
            season_stats,
            x="join_season",
            y="cancellation_probability",
            title="Avg Cancellation by Join Season",
            color="join_season",
            color_discrete_map={
                "❄️ New Year Resolutioner": "#4ECDC4",
                "☀️ Summer Seeker": "#FFA500",
                "📅 Normal Joiner": "#FF6B6B",
            },
        )
        fig.update_layout(
            xaxis_title="Join Season",
            yaxis_title="Avg Cancellation Probability (%)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_intervention_table(df):
    st.subheader("Members Needing Intervention")
    high_risk_df = df[df["risk_level"].isin(["🔴 Critical Risk", "🟠 High Risk"])]
    high_risk_df = high_risk_df.sort_values("cancellation_probability", ascending=False)

    display_cols = member_display_columns(df)

    if high_risk_df.empty:
        st.info("🎉 No high-risk members found!")
        return

    st.dataframe(high_risk_df[display_cols], use_container_width=True, hide_index=True)

    if st.button("💸 Generate Retention Campaign", type="primary"):
        campaign_df = high_risk_df.copy()

        def choose_offer(row):
            if row["risk_level"] == "🔴 Critical Risk":
                if row["visits_per_week"] < 1:
                    return "35% off + free trainer check-in"
                return "25% off next month"
            if row["payment_delay_days"] >= 14:
                return "Payment plan + 15% off"
            return "15% off next month"

        def choose_action(row):
            if row["visits_per_week"] < 1:
                return "Call within 24h and book a comeback session"
            if row["payment_delay_days"] >= 14:
                return "Offer flexible payment help"
            if row["months_since_joined"] < 3:
                return "Send onboarding reset and class invite"
            return "Send personalized retention email"

        def estimate_discount_cost(offer):
            if offer.startswith("35%"):
                return 21
            if offer.startswith("25%"):
                return 15
            if offer.startswith("Payment"):
                return 9
            return 8

        campaign_df["recommended_offer"] = campaign_df.apply(choose_offer, axis=1)
        campaign_df["next_action"] = campaign_df.apply(choose_action, axis=1)
        campaign_df["estimated_discount_cost"] = campaign_df["recommended_offer"].apply(estimate_discount_cost)
        campaign_df["estimated_monthly_value_saved"] = np.where(
            campaign_df["risk_level"] == "🔴 Critical Risk",
            50,
            30,
        )
        campaign_df["estimated_net_value"] = (
            campaign_df["estimated_monthly_value_saved"]
            - campaign_df["estimated_discount_cost"]
        )

        critical_count = len(campaign_df[campaign_df["risk_level"] == "🔴 Critical Risk"])
        high_count = len(campaign_df[campaign_df["risk_level"] == "🟠 High Risk"])
        total_cost = int(campaign_df["estimated_discount_cost"].sum())
        total_saved = int(campaign_df["estimated_monthly_value_saved"].sum())
        net_value = int(campaign_df["estimated_net_value"].sum())

        st.success("✅ Retention campaign generated!")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 Critical Targets", critical_count)
        c2.metric("🟠 High-Risk Targets", high_count)
        c3.metric("💸 Est. Discount Cost", f"€{total_cost}")
        c4.metric("📈 Est. Net Value", f"€{net_value}")

        st.caption(
            f"Estimated monthly value protected: €{total_saved}. "
            "Prioritize calls for members with low visits, then handle payment friction."
        )

        with st.expander("How discounts are assigned"):
            st.markdown(
                """
                - **35% off + free trainer check-in**: critical-risk members visiting less than once per week.
                - **25% off next month**: other critical-risk members who need a strong save attempt.
                - **Payment plan + 15% off**: high-risk members with payment delays of 14+ days.
                - **15% off next month**: other high-risk members who need a lighter retention nudge.
                """
            )

        campaign_cols = [
            col
            for col in [
                "member_id",
                "risk_level",
                "cancellation_probability",
                "visits_per_week",
                "payment_delay_days",
                "recommended_offer",
                "next_action",
                "estimated_discount_cost",
                "estimated_net_value",
            ]
            if col in campaign_df.columns
        ]
        st.dataframe(campaign_df[campaign_cols], use_container_width=True, hide_index=True)

        st.download_button(
            label="⬇️ Download campaign CSV",
            data=to_download_csv(campaign_df[campaign_cols]),
            file_name="retention_campaign.csv",
            mime="text/csv",
        )


def render_detail_sections(df):
    display_cols = member_display_columns(df)

    with st.expander("📊 View Season Breakdown"):
        member_count = "member_id" if "member_id" in df.columns else "join_month"
        season_summary = (
            df.groupby("join_season")
            .agg({"cancellation_probability": "mean", member_count: "count"})
            .round(1)
            .reset_index()
        )
        season_summary.columns = ["Join Season", "Avg Cancellation %", "Member Count"]
        st.dataframe(season_summary, use_container_width=True, hide_index=True)

    with st.expander("🟢 Low and Medium Risk Members (No action needed)"):
        low_risk_df = df[df["risk_level"].isin(["🟡 Medium Risk", "🟢 Low Risk"])]
        if low_risk_df.empty:
            st.write("No medium or low-risk members found.")
        else:
            st.dataframe(low_risk_df[display_cols], use_container_width=True, hide_index=True)


def render_model_comparison(base_df, model_configs):
    rows = []
    comparison_df = base_df.copy()

    for name, config in model_configs.items():
        model = load_model(config)
        probabilities = predict_cancellation(model, base_df)
        comparison_df[f"{name} %"] = np.round(probabilities, 1)
        rows.append({
            "Model": name,
            "Avg Cancellation %": round(float(np.mean(probabilities)), 1),
            "High Risk Members": int(np.sum(probabilities > 50)),
            "Critical Risk Members": int(np.sum(probabilities > 70)),
        })

    summary = pd.DataFrame(rows)
    st.subheader("Algorithm Comparison")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    long_summary = summary.melt(
        id_vars="Model",
        value_vars=["Avg Cancellation %", "High Risk Members", "Critical Risk Members"],
        var_name="Metric",
        value_name="Value",
    )
    fig = px.bar(
        long_summary,
        x="Model",
        y="Value",
        color="Metric",
        barmode="group",
        title="Model Output Comparison",
        color_discrete_sequence=["#4ECDC4", "#FF6B6B", "#FFA500"],
    )
    st.plotly_chart(fig, use_container_width=True)

    model_percent_cols = [f"{name} %" for name in model_configs]
    comparison_df["Average %"] = comparison_df[model_percent_cols].mean(axis=1).round(1)
    comparison_df["Model Spread"] = (
        comparison_df[model_percent_cols].max(axis=1)
        - comparison_df[model_percent_cols].min(axis=1)
    ).round(1)

    display_cols = [
        col
        for col in [
            "member_id",
            "age",
            "months_since_joined",
            "join_month",
            "visits_per_week",
            "payment_delay_days",
            *model_percent_cols,
            "Average %",
            "Model Spread",
        ]
        if col in comparison_df.columns
    ]

    st.subheader("🔎 Member-Level Model Agreement")
    st.dataframe(
        comparison_df.sort_values("Model Spread", ascending=False)[display_cols],
        use_container_width=True,
        hide_index=True,
    )


available = available_models()
if not available:
    st.error("No model files found. Run train_model.py first.")
    st.stop()

st.title("🏋️ KRONOS")
st.subheader("AI-Powered Member Retention")

st.sidebar.header("📁 Upload Member Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

sample_path = Path("gym_members.csv")
use_sample = st.sidebar.checkbox(
    "Use sample gym_members.csv",
    value=sample_path.exists() and uploaded_file is None,
    disabled=not sample_path.exists(),
)

mode_options = list(available.keys())
if len(available) > 1:
    mode_options.append("Compare All Models")

selected_mode = st.sidebar.selectbox("Algorithm", mode_options)
render_model_cards(available, selected_mode)

if uploaded_file:
    source_df = pd.read_csv(uploaded_file)
elif use_sample and sample_path.exists():
    source_df = pd.read_csv(sample_path)
else:
    st.info("👈 Upload a CSV file or use the sample gym_members.csv file.")
    with st.expander("📋 Expected CSV Format"):
        st.markdown(
            """
            Required columns:
            - `age`
            - `months_since_joined`
            - `join_month`
            - `visits_per_week`
            - `payment_delay_days`

            Optional column:
            - `member_id`
            """
        )
    st.stop()

missing = [col for col in REQUIRED_COLUMNS if col not in source_df.columns]
if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.info(f"Required columns: {REQUIRED_COLUMNS}")
    st.stop()

if selected_mode == "Compare All Models":
    algorithm_models = {
        name: config
        for name, config in available.items()
        if name != "Best Default"
    }
    render_model_comparison(source_df, algorithm_models)
    st.stop()

selected_model_path = available[selected_mode]
model = load_model(selected_model_path)
probs = predict_cancellation(model, source_df)
df = add_prediction_columns(source_df, probs, selected_mode)

render_dashboard(df)
render_intervention_table(df)
render_detail_sections(df)

results_csv = to_download_csv(df)
st.download_button(
    label="⬇️ Download results CSV",
    data=results_csv,
    file_name="results.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("#### IART Project 2")
st.caption("T09G07 · KRONOS (AI-Powered Gym Member Retention)")

team_col1, team_col2, team_col3 = st.columns(3)
team_col1.markdown("**Catarina Guimarães**  \nup202307420")
team_col2.markdown("**Sara García**  \nup202306877")
team_col3.markdown("**Stavros Piperakis**  \nup202512352")
