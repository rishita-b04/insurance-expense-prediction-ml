import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -------------------------------------------------
# PAGE SETTINGS
# -------------------------------------------------

st.set_page_config(
    page_title="Insurance Expense Predictor",
    page_icon="🏥",
    layout="wide"
)


# -------------------------------------------------
# CUSTOM DESIGN
# -------------------------------------------------

st.markdown(
    """
    <style>

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, #dff7f3 0%, transparent 28%),
            linear-gradient(135deg, #f7fbff 0%, #eef5fa 100%);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3 {
        color: #123c55 !important;
    }

    [data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.96);
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #d7e6ec;
        box-shadow: 0 10px 28px rgba(24, 68, 88, 0.10);
    }

    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 12px;
        color: white !important;
        font-size: 16px;
        font-weight: 700;
        background: linear-gradient(90deg, #0d6b78, #1597a5);
        box-shadow: 0 7px 16px rgba(13, 107, 120, 0.22);
    }

    [data-testid="stFormSubmitButton"] button:hover {
        border: none !important;
        color: white !important;
        background: linear-gradient(90deg, #095965, #0d8190);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #103b52, #0d6b78);
        padding: 22px;
        border-radius: 17px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 10px 24px rgba(16, 59, 82, 0.20);
    }

    [data-testid="stMetricLabel"] {
        color: #cfeef1 !important;
    }

    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 33px !important;
        font-weight: 800 !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: none;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #f5fafc;
    }

    div[data-baseweb="select"] > div {
        background-color: #f5fafc;
    }

    .hero-box {
        background: linear-gradient(115deg, #103b52, #0d6b78);
        padding: 28px 30px;
        border-radius: 20px;
        margin-bottom: 26px;
        box-shadow: 0 12px 28px rgba(16, 59, 82, 0.18);
    }

    .hero-label {
        color: #7fe0dd;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .hero-title {
        color: white;
        font-size: 36px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #d6edf2;
        font-size: 16px;
    }

    .status-pill {
        display: inline-block;
        margin-top: 16px;
        padding: 6px 12px;
        background-color: rgba(103, 232, 203, 0.14);
        color: #78f0cf;
        border-radius: 18px;
        font-size: 13px;
        font-weight: 700;
    }

    .summary-card {
        background-color: white;
        padding: 18px;
        border-radius: 15px;
        border: 1px solid #dce9ee;
        box-shadow: 0 7px 18px rgba(27, 70, 90, 0.08);
        margin-bottom: 10px;
    }

    .summary-label {
        color: #657b86;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .summary-value {
        color: #123c55;
        font-size: 20px;
        font-weight: 750;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

with open("insurance_expense_model.pkl", "rb") as file:
    model = pickle.load(file)
data = pd.read_csv("insurance.csv")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-label">MACHINE LEARNING HEALTH ANALYTICS</div>
        <div class="hero-title">🏥 Insurance Expense Intelligence</div>
        <div class="hero-subtitle">
            Predict estimated annual medical insurance expenses using
            customer demographic, health and lifestyle information.
        </div>
        <div class="status-pill">● XGBOOST MODEL ONLINE</div>
    </div>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------
# DATASET KPI CARDS
# -------------------------------------------------

total_customers = len(data)
average_expense = data["expenses"].mean()
average_bmi = data["bmi"].mean()
smoker_percentage = data["smoker"].eq("yes").mean() * 100

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="Total Customers",
        value=f"{total_customers:,}"
    )

with kpi2:
    st.metric(
        label="Average Expense",
        value=f"${average_expense:,.0f}"
    )

with kpi3:
    st.metric(
        label="Average BMI",
        value=f"{average_bmi:.1f}"
    )

with kpi4:
    st.metric(
        label="Smokers",
        value=f"{smoker_percentage:.1f}%"
    )

st.write("")

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------

form_column, result_column = st.columns([1.75, 1], gap="large")


# -------------------------------------------------
# CUSTOMER FORM
# -------------------------------------------------

with form_column:

    st.subheader("👤 Customer Information")

    with st.form("insurance_prediction_form"):

        first_column, second_column = st.columns(2)

        with first_column:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25,
                step=1
            )

            gender = st.selectbox(
                "Gender",
                ["female", "male"]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=24.5,
                step=0.1,
                format="%.1f"
            )

        with second_column:

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )

            smoker = st.selectbox(
                "Smoking Status",
                ["no", "yes"]
            )

            region = st.selectbox(
                "Region",
                [
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest"
                ]
            )

        predict_button = st.form_submit_button(
            "Predict Insurance Expense",
            use_container_width=True
        )


# -------------------------------------------------
# DEFAULT OUTPUT
# -------------------------------------------------

with result_column:

    st.subheader("💰 Prediction Result")

    if not predict_button:
        st.info(
            "Enter customer details and click the prediction button."
        )


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if predict_button:

    # Same encoding used during model training
    is_female = 1 if gender == "female" else 0
    is_smoker = 1 if smoker == "yes" else 0
    region_southeast = 1 if region == "southeast" else 0
    bmi_category_obese = 1 if bmi >= 30 else 0

    input_data = pd.DataFrame(
        [[
            age,
            is_female,
            bmi,
            children,
            is_smoker,
            region_southeast,
            bmi_category_obese
        ]],
        columns=[
            "age",
            "is_female",
            "bmi",
            "children",
            "is_smoker",
            "region_southeast",
            "bmi_category_obese"
        ]
    )
    try:

        prediction = float(model.predict(input_data)[0])

        if prediction < 15000:
            expense_level = "Low"
            expense_message = "🟢 Low Estimated Expense"
        elif prediction < 30000:
            expense_level = "Moderate"
            expense_message = "🟡 Moderate Estimated Expense"
        else:
            expense_level = "High"
            expense_message = "🔴 High Estimated Expense"

        # Prediction explanation
        reasons = []

        if smoker == "yes":
            reasons.append(
                "🚬 Smoking significantly increases insurance expenses.")
        else:
            reasons.append("✅ Non-smoking helps reduce insurance costs.")

        if bmi < 18.5:
            reasons.append("⚠️ BMI is underweight.")
        elif bmi < 25:
            reasons.append("💚 BMI is in the healthy range.")
        elif bmi < 30:
            reasons.append("🟡 BMI is overweight.")
        else:
            reasons.append(
                "🔴 High BMI increases health risk and insurance expenses.")

        if age >= 50:
            reasons.append(
                "👴 Higher age generally increases insurance premiums.")
        elif age >= 30:
            reasons.append("🧑 Age has a moderate impact on premium.")
        else:
            reasons.append("🎉 Young age generally lowers insurance cost.")

        if children >= 3:
            reasons.append(
                "👨‍👩‍👧‍👦 More dependents may slightly increase expenses.")
        else:
            reasons.append(
                "👤 Fewer dependents keep insurance relatively lower.")

        with result_column:
            st.success("Prediction completed successfully")

            st.metric(
                label="Estimated Annual Insurance Expense",
                value=f"${prediction:,.2f}"
            )

            if expense_level == "Low":
                st.success(expense_message)
            elif expense_level == "Moderate":
                st.warning(expense_message)
            else:
                st.error(expense_message)

            gauge_value = min(max(prediction, 0), 60000)

            gauge_chart = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=gauge_value,
                    number={
                        "prefix": "$",
                        "valueformat": ",.0f"
                    },
                    title={
                        "text": "Insurance Expense Meter",
                        "font": {"size": 18}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 60000],
                            "tickprefix": "$"
                        },
                        "bar": {"color": "#0d6b78"},
                        "steps": [
                            {"range": [0, 15000], "color": "#d7f4e8"},
                            {"range": [15000, 30000], "color": "#fff0bd"},
                            {"range": [30000, 60000], "color": "#ffd9d9"}
                        ],
                        "threshold": {
                            "line": {"color": "#123c55", "width": 4},
                            "thickness": 0.75,
                            "value": gauge_value
                        }
                    }
                )
            )

            gauge_chart.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=55, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#123c55"}
            )

            st.plotly_chart(
                gauge_chart,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            st.markdown("### 📋 Why this prediction?")
            for reason in reasons:
                st.info(reason)

    except Exception as error:
        with result_column:
            st.error(f"Prediction error: {error}")


# -------------------------------------------------
# CUSTOMER SUMMARY
# -------------------------------------------------

if predict_button:

    st.divider()
    st.subheader("📋 Customer Health and Risk Summary")

    if bmi < 18.5:
        bmi_status = "Underweight"
    elif bmi < 25:
        bmi_status = "Normal"
    elif bmi < 30:
        bmi_status = "Overweight"
    else:
        bmi_status = "Obese"

    summary_1, summary_2, summary_3, summary_4 = st.columns(4)

    with summary_1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Customer Age</div>
                <div class="summary-value">{age} years</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_2:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">BMI Status</div>
                <div class="summary-value">{bmi_status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_3:
        smoking_text = "Smoker" if smoker == "yes" else "Non-smoker"

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Lifestyle Status</div>
                <div class="summary-value">{smoking_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_4:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Dependents</div>
                <div class="summary-value">{children}</div>
            </div>
            """,
            unsafe_allow_html=True


        )

    report_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "BMI": [bmi],
            "Children": [children],
            "Smoker": [smoker],
            "Region": [region],
            "Predicted Expense": [round(prediction, 2)],
            "Expense Level": [expense_level]
        }
    )

    report_csv = report_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=report_csv,
        file_name="insurance_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )


if predict_button:

    st.subheader("📊 BMI Position")

    bmi_chart = go.Figure()

    bmi_chart.add_trace(
        go.Bar(
            x=["Underweight", "Normal", "Overweight", "Obese"],
            y=[18.5, 25, 30, 40],
            text=["Below 18.5", "18.5–24.9", "25–29.9", "30+"],
            textposition="outside",
            marker_color=[
                "#8dc5dd",
                "#72c9a8",
                "#e8c96b",
                "#e88989"
            ],
            hovertemplate="%{x}<extra></extra>"
        )
    )

    bmi_chart.add_hline(
        y=bmi,
        line_dash="dash",
        line_color="#123c55",
        annotation_text=f"Customer BMI: {bmi:.1f}",
        annotation_position="top left"
    )

    bmi_chart.update_layout(
        height=350,
        showlegend=False,
        xaxis_title="BMI Category",
        yaxis_title="BMI Value",
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=20
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)"
    )

    st.plotly_chart(
        bmi_chart,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# -------------------------------------------------
# DATASET INSIGHTS
# -------------------------------------------------

st.divider()
st.subheader("📊 Dataset Insights")

chart_col1, chart_col2 = st.columns(2)


# Smoker vs Non-Smoker Average Expense
smoker_expense = (
    data.groupby("smoker")["expenses"]
    .mean()
    .reset_index()
)

smoker_chart = go.Figure(
    go.Bar(
        x=smoker_expense["smoker"],
        y=smoker_expense["expenses"],
        text=smoker_expense["expenses"],
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        marker_color=["#58b8b0", "#e57f7f"]
    )
)

smoker_chart.update_layout(
    title="Average Expense by Smoking Status",
    xaxis_title="Smoking Status",
    yaxis_title="Average Expense",
    height=360,
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.65)"
)

with chart_col1:
    st.plotly_chart(
        smoker_chart,
        use_container_width=True,
        config={"displayModeBar": False}
    )


# Region-wise Average Expense
region_expense = (
    data.groupby("region")["expenses"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

region_chart = go.Figure(
    go.Bar(
        x=region_expense["region"],
        y=region_expense["expenses"],
        text=region_expense["expenses"],
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        marker_color="#1597a5"
    )
)

region_chart.update_layout(
    title="Average Expense by Region",
    xaxis_title="Region",
    yaxis_title="Average Expense",
    height=360,
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.65)"
)

with chart_col2:
    st.plotly_chart(
        region_chart,
        use_container_width=True,
        config={"displayModeBar": False}
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Model: XGBoost Regressor | "
    "This application is an educational machine-learning portfolio project. "
    "The predicted amount is not an actual insurance quotation."
)
