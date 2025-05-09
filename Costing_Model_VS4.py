import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Refined LDPE Pricing Simulator", layout="centered")
st.title("Refined LDPE Pricing and Margin Simulator with Fixed and Variable Cost Breakdown")

st.markdown("""
This app uses **Monte Carlo simulation** with **Triangular Distributions** to model detailed **fixed and variable cost structures** for LDPE production, supporting expert-level financial modeling.
""")

n_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

st.header("Selling Price Range")
price_min = st.number_input("Minimum Selling Price ($/ton)", value=1900)
price_mode = st.number_input("Most Likely Selling Price ($/ton)", value=2200)
price_max = st.number_input("Maximum Selling Price ($/ton)", value=2400)

st.header("Define Fixed and Variable Components for Each Cost")
def get_cost_inputs(label, fixed_default, var_min_default, var_mode_default, var_max_default):
    st.subheader(label)
    fixed = st.number_input(f"Fixed {label} ($/ton)", value=fixed_default)
    var_min = st.number_input(f"Variable {label} Min ($/ton)", value=var_min_default)
    var_mode = st.number_input(f"Variable {label} Mode ($/ton)", value=var_mode_default)
    var_max = st.number_input(f"Variable {label} Max ($/ton)", value=var_max_default)
    return fixed, var_min, var_mode, var_max

cost_elements = {
    "Resin Cost": (0, 1400, 1500, 1600),
    "Energy Cost": (50, 100, 150, 200),
    "Chemicals/Additives Cost": (10, 20, 30, 40),
    "Labor Cost": (75, 0, 0, 0),
    "Maintenance Cost": (50, 0, 0, 0),
    "Depreciation Cost": (75, 0, 0, 0),
    "Overhead/Administration Cost": (40, 0, 0, 0),
    "Packaging Cost": (5, 10, 20, 30),
    "Logistics Cost": (20, 30, 40, 50)
}

cost_inputs = {}
for label, defaults in cost_elements.items():
    cost_inputs[label] = get_cost_inputs(label, *defaults)

if st.button("Run Simulation"):
    try:
        # Simulate selling prices
        price_sim = np.random.triangular(price_min, price_mode, price_max, n_simulations)

        # Simulate each cost component
        cost_results = {}
        total_fixed = np.zeros(n_simulations)
        total_variable = np.zeros(n_simulations)

        for label, (fixed, var_min, var_mode, var_max) in cost_inputs.items():
            variable_sim = np.random.triangular(var_min, var_mode, var_max, n_simulations) if var_max > 0 else np.zeros(n_simulations)
            total_fixed += fixed
            total_variable += variable_sim
            cost_results[label] = fixed + variable_sim

        total_cost_sim = total_fixed + total_variable
        gross_margin = price_sim - total_cost_sim

        # Build DataFrame
        df_sim = pd.DataFrame({
            'Selling Price': price_sim,
            **{label: values for label, values in cost_results.items()},
            'Total Fixed Cost': total_fixed,
            'Total Variable Cost': total_variable,
            'Total Cost': total_cost_sim,
            'Gross Margin': gross_margin
        })

        st.subheader("Simulation Summary")
        st.write(df_sim.describe())

        st.subheader("Sample Results (First 100 Simulations)")
        st.write(df_sim.head(100))

        st.subheader("Gross Margin Distribution")
        st.bar_chart(df_sim['Gross Margin'].round().value_counts().sort_index())

        # Generate report
        report_text = f"""
        ### Refined LDPE Pricing and Margin Simulation Report

        **Number of Simulations**: {n_simulations}

        **Selling Price Range**: ${price_min} - ${price_max} (Mode: ${price_mode})

        **Cost Component Definitions:**
        {', '.join([f'{label}: Fixed ${vals[0]}/ton, Variable ${vals[1]}-${vals[3]}/ton' for label, vals in cost_inputs.items()])}

        **Simulation Summary Statistics:**

        {df_sim.describe()}
        """

        st.subheader("Simulation Report")
        st.markdown(report_text)

        # Prepare file for download
        towrite = BytesIO()
        df_sim.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button(
            label="📥 Download Simulation Results as CSV",
            data=towrite,
            file_name="refined_ldpe_pricing_simulation_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
