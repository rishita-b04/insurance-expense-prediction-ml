from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Insurance Expense Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "insurance_expense_model.pkl"
DATA_PATH = BASE_DIR / "insurance.csv"

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f7fbff 0%, #e9f8f5 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #103b52 0%, #0d6b78 100%);
}
[data-testid="stSidebar"] * { color: white; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #103b52, #0d6b78);
    padding: 20px; border-radius: 16px;
    box-shadow: 0 8px 20px rgba(16,59,82,.18);
}
[data-testid="stMetricLabel"] { color: #cfeef1 !important; }
[data-testid="stMetricValue"] { color: white !important; font-weight: 800 !important; }
[data-testid="stForm"] {
    background: white; padding: 24px; border-radius: 18px;
    border: 1px solid #d7e6ec; box-shadow: 0 8px 22px rgba(27,70,90,.09);
}
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    width: 100%; height: 48px; border: none; border-radius: 12px;
    color: white !important; font-weight: 700;
    background: linear-gradient(90deg, #0d6b78, #1597a5);
}
.hero {
    background: linear-gradient(135deg, #103b52, #0d6b78 58%, #1597a5);
    padding: 42px; border-radius: 24px; color: white;
    box-shadow: 0 16px 36px rgba(16,59,82,.22); margin-bottom: 28px;
}
.hero-label { color: #8ce8df; font-size: 13px; font-weight: 750; letter-spacing: 2px; }
.hero-title { color: white; font-size: 44px; font-weight: 850; line-height: 1.15; margin: 10px 0; }
.hero-text { color: #e1f2f5; font-size: 16px; line-height: 1.7; max-width: 850px; }
.tag {
    display: inline-block; margin: 18px 8px 0 0; padding: 8px 14px;
    border-radius: 20px; background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.24); font-weight: 700; font-size: 13px;
}
.summary-card {
    background: white; padding: 18px; border-radius: 15px;
    border: 1px solid #dce9ee; box-shadow: 0 7px 18px rgba(27,70,90,.08);
}
.summary-label { color: #657b86; font-size: 13px; }
.summary-value { color: #123c55; font-size: 20px; font-weight: 750; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)

try:
    data = load_data()
    data = data.rename(columns={"charges": "expenses"})
    data_error = None
except Exception as error:
    data = None
    data_error = str(error)

with st.sidebar:
    st.title("🏥 Insurance AI")
    st.caption("Machine Learning Expense Prediction")
    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠 Home", "💰 Prediction", "📊 Analytics", "ℹ️ About Project"],
    )
    st.divider()
    if model is not None:
        st.success("● XGBoost Model Online")
    else:
        st.error("● Model Unavailable")
    st.caption("Developed by Rishita Bagri")

if page == "🏠 Home":
    st.markdown(
        """
<div class="hero">
<div class="hero-label">MACHINE LEARNING HEALTH ANALYTICS</div>
<div class="hero-title">🏥 Insurance Expense Intelligence</div>
<div class="hero-text">
Estimate annual medical insurance expenses using customer demographic,
health and lifestyle information. Explore predictions, analytics and model insights.
</div>
<div>
<span class="tag">🤖 XGBoost Model</span>
<span class="tag">📊 Interactive Analytics</span>
<span class="tag">₹ INR + $ USD</span>
<span class="tag">📥 Download Report</span>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        st.markdown("# 🏥")
        st.markdown("## Health Insurance Analytics")
        st.markdown("### 👤 Customer Information")
        st.markdown("### 🤖 Machine Learning Model")
        st.markdown("### 📊 Interactive Insights")
    with right:
        st.subheader("Smart Insurance Estimation")
        st.write(
            "This application uses an XGBoost regression model to estimate annual "
            "medical insurance expenses in USD and approximate INR."
        )
        st.write("#### Main Features")
        st.write("✅ Insurance expense prediction")
        st.write("✅ INR and USD output")
        st.write("✅ Expense risk category")
        st.write("✅ Interactive analytics")
        st.write("✅ Downloadable report")

    st.divider()
    st.subheader("📌 Project Highlights")
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        with st.container(border=True):
            st.markdown("### 🤖 XGBoost")
            st.write("Final regression model for annual expense estimation.")
    with h2:
        with st.container(border=True):
            st.markdown("### 📊 Analytics")
            st.write("Smoking, BMI, age and region-based insights.")
    with h3:
        with st.container(border=True):
            st.markdown("### ₹ INR + $ USD")
            st.write("Results shown in both currencies.")
    with h4:
        with st.container(border=True):
            st.markdown("### 📥 Report")
            st.write("Download prediction details as CSV.")

    st.divider()
    st.subheader("⚙️ How It Works")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.info("1️⃣ Enter customer details")
    with s2:
        st.info("2️⃣ Process model features")
    with s3:
        st.info("3️⃣ Generate prediction")
    with s4:
        st.info("4️⃣ Review and download")

elif page == "💰 Prediction":
    st.markdown(
        """
<div class="hero">
<div class="hero-label">SMART INSURANCE ESTIMATION</div>
<div class="hero-title">💰 Predict Insurance Expense</div>
<div class="hero-text">Enter customer information to estimate annual medical insurance expenses.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    usd_to_inr = st.number_input(
        "USD to INR conversion rate", 1.0, 200.0, 87.0, 0.5,
        help="This is a manual approximate conversion rate.",
    )

    form_col, result_col = st.columns([1.65, 1], gap="large")
    with form_col:
        st.subheader("👤 Customer Information")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", 18, 100, 25, 1)
                gender = st.selectbox("Gender", ["female", "male"])
                bmi = st.number_input(
                    "BMI", 10.0, 60.0, 24.5, 0.1, format="%.1f")
            with c2:
                children = st.number_input("Number of Children", 0, 10, 0, 1)
                smoker = st.selectbox("Smoking Status", ["no", "yes"])
                region = st.selectbox(
                    "Region", ["northeast", "northwest",
                               "southeast", "southwest"]
                )
            predict_button = st.form_submit_button(
                "Predict Insurance Expense", use_container_width=True
            )

    with result_col:
        st.subheader("🎯 Prediction Result")
        if not predict_button:
            st.info("Complete the form and click the prediction button.")

    if predict_button:
        input_data = pd.DataFrame(
            [[
                age,
                1 if gender == "female" else 0,
                bmi,
                children,
                1 if smoker == "yes" else 0,
                1 if region == "southeast" else 0,
                1 if bmi >= 30 else 0,
            ]],
            columns=[
                "age", "is_female", "bmi", "children", "is_smoker",
                "region_southeast", "bmi_category_obese",
            ],
        )

        try:
            prediction_usd = float(model.predict(input_data)[0])
            prediction_inr = prediction_usd * usd_to_inr

            if prediction_usd < 15000:
                level, message = "Low", "🟢 Low Estimated Expense"
            elif prediction_usd < 30000:
                level, message = "Moderate", "🟡 Moderate Estimated Expense"
            else:
                level, message = "High", "🔴 High Estimated Expense"

            reasons = []
            reasons.append(
                "🚬 Smoking significantly increases insurance expenses."
                if smoker == "yes" else
                "✅ Non-smoking helps reduce insurance costs."
            )
            if bmi < 18.5:
                bmi_status = "Underweight"
                reasons.append("⚠️ BMI is below the normal range.")
            elif bmi < 25:
                bmi_status = "Normal"
                reasons.append("💚 BMI is in the healthy range.")
            elif bmi < 30:
                bmi_status = "Overweight"
                reasons.append("🟡 BMI is in the overweight range.")
            else:
                bmi_status = "Obese"
                reasons.append(
                    "🔴 High BMI can increase health risk and expense.")
            reasons.append(
                "👴 Higher age generally increases insurance premiums."
                if age >= 50 else
                "🧑 Age has a moderate influence on the prediction."
                if age >= 30 else
                "🎉 Younger age generally helps keep expenses lower."
            )
            reasons.append(
                "👨‍👩‍👧‍👦 More dependents may slightly increase expenses."
                if children >= 3 else
                "👤 Fewer dependents keep expenses relatively lower."
            )

            with result_col:
                st.success("Prediction completed successfully")
                st.metric("Estimated Expense in INR",
                          f"₹{prediction_inr:,.0f}")
                st.metric("Original Model Output in USD",
                          f"${prediction_usd:,.2f}")
                if level == "Low":
                    st.success(message)
                elif level == "Moderate":
                    st.warning(message)
                else:
                    st.error(message)

                gauge_value = min(max(prediction_usd, 0), 60000)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=gauge_value,
                    number={"prefix": "$", "valueformat": ",.0f"},
                    title={"text": "Insurance Expense Meter"},
                    gauge={
                        "axis": {"range": [0, 60000], "tickprefix": "$"},
                        "bar": {"color": "#0d6b78"},
                        "steps": [
                            {"range": [0, 15000], "color": "#d7f4e8"},
                            {"range": [15000, 30000], "color": "#fff0bd"},
                            {"range": [30000, 60000], "color": "#ffd9d9"},
                        ],
                    },
                ))
                gauge.update_layout(
                    height=285, margin=dict(l=20, r=20, t=55, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(gauge, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown("### 📋 Why this prediction?")
                for reason in reasons:
                    st.info(reason)

            st.divider()
            st.subheader("📋 Customer Health and Risk Summary")
            q1, q2, q3, q4 = st.columns(4)
            items = [
                ("Customer Age", f"{age} years"),
                ("BMI Status", bmi_status),
                ("Lifestyle Status", "Smoker" if smoker == "yes" else "Non-smoker"),
                ("Dependents", str(children)),
            ]
            for col, (label, value) in zip([q1, q2, q3, q4], items):
                with col:
                    st.markdown(
                        f'<div class="summary-card"><div class="summary-label">{label}</div>'
                        f'<div class="summary-value">{value}</div></div>',
                        unsafe_allow_html=True,
                    )

            report = pd.DataFrame({
                "Age": [age], "Gender": [gender], "BMI": [bmi],
                "Children": [children], "Smoker": [smoker], "Region": [region],
                "Predicted Expense USD": [round(prediction_usd, 2)],
                "USD to INR Rate": [usd_to_inr],
                "Approximate Expense INR": [round(prediction_inr, 2)],
                "Expense Level": [level],
            })
            st.download_button(
                "📥 Download Prediction Report",
                report.to_csv(index=False).encode("utf-8"),
                "insurance_prediction_report.csv",
                "text/csv",
                use_container_width=True,
            )
        except Exception as error:
            with result_col:
                st.error(f"Prediction error: {error}")


elif page == "📊 Analytics":
    st.markdown(
        """
<div class="hero">
<div class="hero-label">REAL DATASET INSIGHTS</div>
<div class="hero-title">📊 Insurance Analytics Dashboard</div>
<div class="hero-text">
Explore smoking, BMI and age-wise insurance expense patterns.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if data is None:
        st.error(f"Dataset could not be loaded: {data_error}")
        st.stop()

    required = {"age", "bmi", "children", "smoker", "region", "expenses"}
    missing = required.difference(data.columns)

    if missing:
        st.error(f"Missing columns in insurance.csv: {sorted(missing)}")
        st.stop()

    rate = st.number_input(
        "USD to INR conversion rate for analytics",
        min_value=1.0,
        max_value=200.0,
        value=87.0,
        step=0.5,
    )

    df = data.copy()
    df["expenses_inr"] = df["expenses"] * rate

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric("Total Customers", f"{len(df):,}")

    with k2:
        st.metric("Average Expense", f"₹{df['expenses_inr'].mean():,.0f}")

    with k3:
        st.metric("Average BMI", f"{df['bmi'].mean():.1f}")

    with k4:
        st.metric("Smokers", f"{df['smoker'].eq('yes').mean() * 100:.1f}%")

    st.divider()

    a, b = st.columns(2)

    smoker_avg = (
        df.groupby("smoker", as_index=False)["expenses_inr"]
        .mean()
    )

    fig1 = px.bar(
        smoker_avg,
        x="smoker",
        y="expenses_inr",
        text="expenses_inr",
        title="Average Expense by Smoking Status",
    )
    fig1.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="outside",
    )
    fig1.update_layout(
        xaxis_title="Smoking Status",
        yaxis_title="Expense in INR",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        showlegend=False,
        height=390,
    )

    with a:
        st.plotly_chart(
            fig1,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    age_avg = df.copy()
    age_avg["Age Group"] = pd.cut(
        age_avg["age"],
        bins=[18, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"],
        include_lowest=True,
    )
    age_avg = (
        age_avg.groupby(
            "Age Group",
            as_index=False,
            observed=False,
        )["expenses_inr"]
        .mean()
    )

    fig2 = px.bar(
        age_avg,
        x="Age Group",
        y="expenses_inr",
        text="expenses_inr",
        title="Average Expense by Age Group",
        color="expenses_inr",
        color_continuous_scale="Teal"
    )
    fig2.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="outside",
    )
    fig2.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Expense in INR",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        showlegend=False,
        height=390,
    )

    with b:
        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    c, d = st.columns(2)

    fig3 = px.scatter(
        df,
        x="bmi",
        y="expenses_inr",
        color="smoker",
        size="age",
        hover_data=["age", "region", "children"],
        title="BMI vs Insurance Expense"
    )
    fig3.update_layout(
        yaxis_title="Expense in INR",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        height=410,
    )

    with c:
        st.plotly_chart(
            fig3,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    correlation_data = df[
        ["age", "bmi", "children", "expenses_inr"]
    ].rename(
        columns={
            "age": "Age",
            "bmi": "BMI",
            "children": "Children",
            "expenses_inr": "Expense",
        }
    )
    correlation_matrix = correlation_data.corr()

    fig4 = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        color_continuous_scale="Teal",
        aspect="auto",
        title="Feature Correlation Heatmap",
    )
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=430,
    )

    with d:
        st.plotly_chart(
            fig4,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.subheader("🔍 Key Insights")
    st.success("Smokers have substantially higher average insurance expenses.")
    st.info("Average insurance expense increases across older age groups.")
    st.warning(
        "BMI and age show stronger relationships with expense than children.")


elif page == "ℹ️ About Project":
    st.markdown(
        """
<div class="hero">
<div class="hero-label">PROJECT DOCUMENTATION</div>
<div class="hero-title">ℹ️ About Insurance Expense Intelligence</div>
<div class="hero-text">
A machine-learning portfolio project developed to estimate annual medical insurance expenses.
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)

    with a1:
        with st.container(border=True):
            st.subheader("🎯 Project Objective")
            st.write(
                "Predict annual medical insurance expenses using customer "
                "health and lifestyle data."
            )

    with a2:
        with st.container(border=True):
            st.subheader("📁 Dataset")
            st.write(
                "The dataset contains 1,338 customer records with original "
                "expenses in USD."
            )

    a3, a4 = st.columns(2)

    with a3:
        with st.container(border=True):
            st.subheader("🤖 Model")
            st.write(
                "XGBoost Regressor is used as the final prediction model."
            )

    with a4:
        with st.container(border=True):
            st.subheader("🛠️ Technologies")
            st.write(
                "Python, Pandas, Plotly, Streamlit, Scikit-learn, "
                "XGBoost and Pickle."
            )

    st.warning(
        "This is an educational portfolio project, not an official "
        "insurance quotation."
    )


st.divider()
st.caption("Insurance Expense Intelligence | Developed by Rishita Bagri")
