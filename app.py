import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="KRONOS - Gym Retention", layout="wide")

# Load model
model_path = Path('churn_model.pkl')
if not model_path.exists():
    st.error("Missing churn_model.pkl. Run train_model.py first to generate the model.")
    st.stop()

model = joblib.load(model_path)

# Header
st.title("🏋️ KRONOS")
st.subheader("AI-Powered Member Retention")

# Sidebar
st.sidebar.header("📁 Upload Member Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

# Season info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 High Risk Join Months")
st.sidebar.markdown("🔴 **January, February** (New Year's Resolutioners)")
st.sidebar.markdown("🔴 **June, July, August** (Summer Beach Body)")
st.sidebar.markdown("🟢 **All other months** (Normal)")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required = ['age', 'months_since_joined', 'join_month',
                'visits_per_week', 'payment_delay_days']

    if all(col in df.columns for col in required):

        # ✅ Fix: use predict_proba to get actual probabilities
        churn_probs = model.predict_proba(df[required])[:, 1] * 100
        df['churn_probability'] = np.round(churn_probs, 1)

        # Risk levels
        def get_risk(prob):
            if prob > 70:
                return "🔴 Critical Risk"
            elif prob > 50:
                return "🟠 High Risk"
            elif prob > 30:
                return "🟡 Medium Risk"
            else:
                return "🟢 Low Risk"

        df['risk_level'] = df['churn_probability'].apply(get_risk)

        # Season label
        def get_season(month):
            if month in [1, 2]:
                return "❄️ New Year Resolutioner"
            elif month in [6, 7, 8]:
                return "☀️ Summer Seeker"
            else:
                return "📅 Normal Joiner"

        df['join_season'] = df['join_month'].apply(get_season)

        # ✅ Fix: define display_cols here, outside any conditional block
        display_cols = [col for col in ['member_id', 'age', 'months_since_joined', 'join_season',
                        'visits_per_week', 'payment_delay_days', 'churn_probability', 'risk_level']
                        if col in df.columns]

        results_csv = df.to_csv(index=False).encode('utf-8')

        # Dashboard metrics
        st.subheader("📊 Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Members", len(df))
        col2.metric("Avg Churn Probability", f"{df['churn_probability'].mean():.1f}%")
        col3.metric("Critical Risk", len(df[df['risk_level'] == "🔴 Critical Risk"]))
        col4.metric("High Risk", len(df[df['risk_level'] == "🟠 High Risk"]))

        risk_summary = df['risk_level'].value_counts().reset_index()
        risk_summary.columns = ['Risk Level', 'Count']
        st.dataframe(risk_summary, use_container_width=True, hide_index=True)

        # Charts
        col_left, col_right = st.columns(2)

        with col_left:
            fig1 = px.histogram(df, x='churn_probability', nbins=20,
                               title="Churn Probability Distribution",
                               color_discrete_sequence=['#FF6B6B'])
            fig1.update_layout(xaxis_title="Churn Probability (%)",
                              yaxis_title="Number of Members")
            st.plotly_chart(fig1, use_container_width=True)

        with col_right:
            season_stats = df.groupby('join_season')['churn_probability'].mean().reset_index()
            fig2 = px.bar(season_stats, x='join_season', y='churn_probability',
                         title="Avg Churn by Join Season",
                         color='join_season',
                         color_discrete_map={
                             '❄️ New Year Resolutioner': '#FF6B6B',
                             '☀️ Summer Seeker': '#FFA500',
                             '📅 Normal Joiner': '#4ECDC4'
                         })
            fig2.update_layout(xaxis_title="Join Season", yaxis_title="Avg Churn Probability (%)")
            st.plotly_chart(fig2, use_container_width=True)

        # High risk members
        st.subheader("🔴 Members Needing Intervention")
        high_risk_df = df[df['risk_level'].isin(["🔴 Critical Risk", "🟠 High Risk"])].sort_values('churn_probability', ascending=False)

        if len(high_risk_df) > 0:
            st.dataframe(high_risk_df[display_cols], use_container_width=True)

            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("💸 Generate Discount Campaign", type="primary"):
                    critical_count = len(high_risk_df[high_risk_df['risk_level'] == "🔴 Critical Risk"])
                    high_count = len(high_risk_df[high_risk_df['risk_level'] == "🟠 High Risk"])
                    st.success(f"""
                    ✅ **Discount Campaign Generated!**
                    - {critical_count} critical risk members: 25% off next month
                    - {high_count} high risk members: 15% off next month
                    - Total potential revenue saved: ~€{critical_count * 50 + high_count * 30}
                    """)
        else:
            st.info("No high-risk members found!")

        # Season breakdown
        with st.expander("📊 View Season Breakdown"):
            member_count = 'member_id' if 'member_id' in df.columns else 'join_month'
            season_summary = df.groupby('join_season').agg({
                'churn_probability': 'mean',
                member_count: 'count'
            }).round(1).reset_index()
            season_summary.columns = ['Join Season', 'Avg Churn %', 'Member Count']
            st.dataframe(season_summary, use_container_width=True)

        # Low risk members
        with st.expander("🟢 Low Risk Members (No action needed)"):
            low_risk_df = df[df['risk_level'] == "🟢 Low Risk"]
            if len(low_risk_df) > 0:
                st.dataframe(low_risk_df[display_cols], use_container_width=True)
            else:
                st.write("No low risk members found")

        st.download_button(
            label="⬇️ Download results CSV",
            data=results_csv,
            file_name="results.csv",
            mime="text/csv",
        )

    else:
        missing = [col for col in required if col not in df.columns]
        st.error(f"❌ Missing columns: {missing}")
        st.info(f"Required columns: {required}")
else:
    st.info("👈 Upload any CSV file with the required member columns to begin")

    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        **Required columns:**
        - `member_id` (optional, for identification)
        - `age` (18-70)
        - `months_since_joined` (0.5-36)
        - `join_month` (1-12, where 1=January)
        - `visits_per_week` (0-7)
        - `payment_delay_days` (0-30)

        **Example:**
        ```
        member_id,age,months_since_joined,join_month,visits_per_week,payment_delay_days
        1,34,12.5,3,4.2,0
        ```
        """)

# Footer
st.markdown("---")
st.markdown("**KRONOS** — Because time tells who stays ⏰")