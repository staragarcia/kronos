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
    "Best Default": {
        "path": Path("churn_model.pkl"),
        "label": "Best Default",
        "note": "The best model selected by train_model.py.",
    },
    "Logistic Regression": {
        "path": Path("churn_logistic_model.pkl"),
        "label": "Logistic Regression",
        "note": "Simple and explainable baseline.",
    },
    "Random Forest": {
        "path": Path("churn_random_forest_model.pkl"),
        "label": "Random Forest",
        "note": "Good at non-linear member behavior patterns.",
    },
    "Gradient Boosting": {
        "path": Path("churn_gradient_boosting_model.pkl"),
        "label": "Gradient Boosting",
        "note": "Sequential tree model for subtle tabular patterns.",
    },
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def available_models():
    return {
        name: config
        for name, config in MODEL_OPTIONS.items()
        if config["path"].exists()
    }


def predict_churn(model, df):
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
    result["churn_probability"] = np.round(probabilities, 1)
    result["risk_level"] = result["churn_probability"].apply(get_risk)
    result["join_season"] = result["join_month"].apply(get_season)
    return result


def render_model_cards(model_configs, selected_model):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Models")
    for name, config in model_configs.items():
        marker = "✨ selected" if name == selected_model else "ready"
        st.sidebar.caption(f"{config['label']} — {marker}")


def render_dashboard(df):
    st.subheader("📊 Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Members", len(df))
    col2.metric("📈 Avg Churn Probability", f"{df['churn_probability'].mean():.1f}%")
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
            x="churn_probability",
            nbins=20,
            title="Churn Probability Distribution",
            color_discrete_sequence=["#FF6B6B"],
        )
        fig.update_layout(
            xaxis_title="Churn Probability (%)",
            yaxis_title="Number of Members",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        season_stats = (
            df.groupby("join_season")["churn_probability"]
            .mean()
            .round(1)
            .reset_index()
        )
        fig = px.bar(
            season_stats,
            x="join_season",
            y="churn_probability",
            title="Avg Churn by Join Season",
            color="join_season",
            color_discrete_map={
                "❄️ New Year Resolutioner": "#FF6B6B",
                "☀️ Summer Seeker": "#FFA500",
                "📅 Normal Joiner": "#4ECDC4",
            },
        )
        fig.update_layout(
            xaxis_title="Join Season",
            yaxis_title="Avg Churn Probability (%)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_intervention_table(df):
    st.subheader("🔴 Members Needing Intervention")
    high_risk_df = df[df["risk_level"].isin(["🔴 Critical Risk", "🟠 High Risk"])]
    high_risk_df = high_risk_df.sort_values("churn_probability", ascending=False)

    display_cols = [
        col
        for col in [
            "member_id",
            "age",
            "months_since_joined",
            "join_season",
            "visits_per_week",
            "payment_delay_days",
            "churn_probability",
            "risk_level",
            "model_used",
        ]
        if col in df.columns
    ]

    if high_risk_df.empty:
        st.info("🎉 No high-risk members found!")
        return

    st.dataframe(high_risk_df[display_cols], use_container_width=True, hide_index=True)

    if st.button("💸 Generate Discount Campaign", type="primary"):
        critical_count = len(high_risk_df[high_risk_df["risk_level"] == "🔴 Critical Risk"])
        high_count = len(high_risk_df[high_risk_df["risk_level"] == "🟠 High Risk"])
        st.success(
            f"✅ Campaign generated! {critical_count} critical-risk members get 25% off, "
            f"{high_count} high-risk members get 15% off. "
            f"Estimated revenue protected: ~€{critical_count * 50 + high_count * 30}."
        )


def render_detail_sections(df):
    display_cols = [
        col
        for col in [
            "member_id",
            "age",
            "months_since_joined",
            "join_season",
            "visits_per_week",
            "payment_delay_days",
            "churn_probability",
            "risk_level",
            "model_used",
        ]
        if col in df.columns
    ]

    with st.expander("📊 View Season Breakdown"):
        member_count = "member_id" if "member_id" in df.columns else "join_month"
        season_summary = (
            df.groupby("join_season")
            .agg({"churn_probability": "mean", member_count: "count"})
            .round(1)
            .reset_index()
        )
        season_summary.columns = ["Join Season", "Avg Churn %", "Member Count"]
        st.dataframe(season_summary, use_container_width=True, hide_index=True)

    with st.expander("🟢 Low Risk Members (No action needed)"):
        low_risk_df = df[df["risk_level"] == "🟢 Low Risk"]
        if low_risk_df.empty:
            st.write("No low-risk members found.")
        else:
            st.dataframe(low_risk_df[display_cols], use_container_width=True, hide_index=True)


def render_model_comparison(base_df, model_configs):
    rows = []
    comparison_df = base_df.copy()

    for name, config in model_configs.items():
        model = load_model(config["path"])
        probabilities = predict_churn(model, base_df)
        comparison_df[f"{name} %"] = np.round(probabilities, 1)
        rows.append({
            "Model": name,
            "Avg Churn %": round(float(np.mean(probabilities)), 1),
            "High Risk Members": int(np.sum(probabilities > 50)),
            "Critical Risk Members": int(np.sum(probabilities > 70)),
        })

    summary = pd.DataFrame(rows)
    st.subheader("🧠 Algorithm Comparison")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    long_summary = summary.melt(
        id_vars="Model",
        value_vars=["Avg Churn %", "High Risk Members", "Critical Risk Members"],
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

selected_mode = st.sidebar.selectbox("🧠 Algorithm", mode_options)
render_model_cards(available, selected_mode)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 High Risk Join Months")
st.sidebar.markdown("🔴 **January, February** (New Year's Resolutioners)")
st.sidebar.markdown("🔴 **June, July, August** (Summer Beach Body)")
st.sidebar.markdown("🟢 **All other months** (Normal)")

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

selected_config = available[selected_mode]
model = load_model(selected_config["path"])
probs = predict_churn(model, source_df)
df = add_prediction_columns(source_df, probs, selected_config["label"])

st.caption(selected_config["note"])
render_dashboard(df)
render_intervention_table(df)
render_detail_sections(df)

results_csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download results CSV",
    data=results_csv,
    file_name="results.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("**KRONOS** — Because time tells who stays ⏰")
