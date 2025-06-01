# import streamlit
import streamlit as st

# import pandas and numpy
import pandas as pd
import numpy as np

# import statsmodels packages for power analysis and sample sizes
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.power import TTestIndPower, zt_ind_solve_power

# import plotly for visualizations
import plotly.express as px
import plotly.graph_objects as go

##### custom functions for sample size and power

# sample size for proportions
def sample_size_proportions(p1, p2, alpha, power, alternative):
    effect_size = proportion_effectsize(p1, p2)
    result = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)
    return round(float(result))

# sample size for means
def sample_size_means(mean1, mean2, std_dev, alpha, power, alternative):
    effect_size = abs(mean2 - mean1) / std_dev
    analysis = TTestIndPower()
    result = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=alternative)
    return round(float(result))

# explanations
def generate_explanation(test_type, sample_size, alpha, power, alternative, delta):
    direction = {
        "two-sided": f"a difference of {abs(delta):.3f}",
        "larger": f"an increase of {abs(delta):.3f}",
        "smaller": f"a decrease of {abs(delta):.3f}"
    }

    return (
        f"To detect {direction[alternative]} in {test_type.lower()} between the two groups "
        f"with **{int(power * 100)}% power** and a **{int(alpha * 100)}% significance level**, "
        f"you need **at least {sample_size} users per group**."
    )

# downloading results
def create_download_df(test_type, sample_size, alpha, power, inputs):
    result = {
        "Test Type": test_type,
        "Baseline": inputs["baseline"],
        "Delta": inputs["delta"],
        "Expected Result (Test Group)": inputs["expected"],
        "Alpha": alpha,
        "Power": power,
        "Sample Size per Arm": sample_size
    }
    return pd.DataFrame([result])

##### the UI for the app

# Title
st.title("Sample Size Estimator")
# subheader
st.subheader("Calculate the sample size required to detect significant differences in group rates or means for your A/B test")

st.divider()

# User Inputs for (1) test type, (2) alpha level, (3) power, and (4) alterantive hypothesis
test_type = st.radio("Select the type of hypothesis test", ["Difference in Group Rates (proportion test)", "Difference in Group Means (t-test)"])
alpha = st.number_input("Significance Level (alpha)", min_value = 0.001, max_value = 0.2, value = 0.05, step = 0.001, format = "%.3f")
power = st.number_input("Statistical Power", min_value = 0.5, max_value = 0.99, value = 0.8, step = 0.01, format = "%.2f")
alternative = st.selectbox("Alternative Hypothesis", options = ["two-sided", "larger", "smaller"],
                           help = "Choose 'two-sided' if the effect could be an increase or a decrease, 'larger' if you expect only an increase, or 'smaller' if you expect only a decrease.")

st.divider()

# Proportion selection
if test_type == "Difference in Group Rates (proportion test)":
    st.subheader("Baseline and Delta Inputs for Difference in Proportions")
    p1 = st.number_input("Baseline Rate (Control Group)", min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.01, format = "%.3f",
                         help = "The rate for the baseline group (e.g., 0.1 means a 10% rate)")
    delta = st.number_input(
        "Expected Lift or Change (must be absolute value)",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.005,
        format="%.3f",
        help="The expected absolute difference in conversion rate (e.g., 0.02 = +2 or -2 percentage points). Enter a positive number. The analysis will apply the correct direction (+ or -) based on the 'Alternative Hypothesis' selection."
    )

    # to get the right proportion for group 2, we need to take into account the delta and type of alternative hypothesis test selected
    if alternative == "smaller":
        p2 = p1 - abs(delta)
    else:
        p2 = p1 + abs(delta)

    # calculation button
    if st.button("Calculate Sample Size"):
        # first do some validation - make sure the inputs are appropriate
        if not (0 <= p2 <= 1):
            st.error("❌ Selected p2 is outside the valid range (0 - 1)")
        elif np.isnan(p1) or np.isnan(p2):
            st.error("❌ Invalid numerical input for one or more proportions")
        # if the validations are passed then run the sample size calculations
        elif abs(delta) < 0.0001:
            st.error("❌ The expected change value is too small to detect meaningfully. Input a larger value.")
        else:
            sample_size = sample_size_proportions(p1, p2, alpha, power, alternative)
            st.success(f"✅ Analysis completed!") # success
            st.metric(label = "Sample size per group", value = sample_size)
            # generate the explanation of the results
            st.info(generate_explanation("Proportions", sample_size, alpha, power, alternative, delta))
            # inputs needed for the results and download
            inputs = {
                "baseline": p1,
                "delta": delta,
                "expected": p2
            }

            # make a dataframe of the results
            df_result = create_download_df("Proportions", sample_size, alpha, power, inputs)

            # make csv for downloading
            csv = df_result.to_csv(index=False).encode("utf-8") # utf-8 should be compatible with most 
            st.download_button(
                label = "Download Result as CSV",
                data = csv,
                file_name = "sample_size_proportions.csv",
                mime = "text/csv"
            )

# Run it for difference in means
elif test_type == "Difference in Group Means (t-test)":
    st.subheader("Baseline and Delta Inputs for Difference in Means")
    # get user inputs and delta
    mean1 = st.number_input("Baseline Mean (Control Group)", value = 50.0, step = 1.0,
                            help = "The average value for your baseline group")
    delta = st.number_input(
        "Expected Lift or Change (must be absolute value)",
        value=5.0,
        step=0.5,
        help="The size of the effect you want to detect. Enter a positive number (e.g., 5 = +5 or -5). The analysis will apply the correct direction based on the 'Alternative Hypothesis' selection."
    )

    # get the right mean based on the alternative parmater - if smaller then we need mean2 to be mean1 - delta
    if alternative == "smaller":
        mean2 = mean1 - abs(delta)
    else:
        mean2 = mean1 + abs(delta)

    # pooled std -- this needs to be provided (use the control group)
    std_dev = st.number_input("Standard Deviation", min_value = 0.01, value = 10.0, step = 0.5) 

    # calculate
    if st.button("Calculate Sample Size"):
        # validation and checks
        if std_dev <= 0:
            st.error("❌ Standard deviation must be positive.")
        elif mean1 == mean2:
            st.error("❌ Mean difference must be non-zero.")
        elif any(np.isnan([mean1, mean2, std_dev])) or any(np.isinf([mean1, mean2, std_dev])):
            st.error("❌ Mean and standard deviation values must be valid numbers.")
        # if validations are passed then run the estimate
        else:
            sample_size = sample_size_means(mean1, mean2, std_dev, alpha, power, alternative)
            st.success(f"✅ Analysis completed!") # success
            st.metric(label = "Sample size per group", value = sample_size)
            st.info(generate_explanation("Means", sample_size, alpha, power, alternative, delta))
            # inputs for the results
            inputs = {
                "baseline": mean1,
                "delta": delta,
                "expected": mean2
            }

            # make dataframe of results
            df_result = create_download_df("Means", sample_size, alpha, power, inputs)
            
            # download the results
            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label = "Download Result as CSV",
                data = csv,
                file_name = "sample_size_means.csv",
                mime = "text/csv"
            )

st.divider()
st.write("Developed by A. C. Sanders | 2025")