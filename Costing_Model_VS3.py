import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="LDPE Pricing Simulator", layout="centered")
st.title("LDPE Pricing and Margin Simulator with Triangular Distribution and Cost Classification")

st.markdown("""
This app uses **Monte Carlo simulation** with **Triangular Distributions** to model variable **selling prices** and **detailed cost structures** (fixed and variable) on your **gross margin**.
""")

n_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

st.header("Selling Price Range")
price_min = st.number_input("Minimum Selling Price ($/ton)", value=1900)
price_mode = st.number_input("Most Likely Selling Price ($/ton)", value=2200)
price_max = st.number_input("Maximum Selling Price ($/ton)", value=2400)

st.header("Define Variable and Fixed Costs")

variable_cost_base = st.number_input("Base Variable Cost ($/ton)", value=1800)
fixed_cost_base = st.number_input("Base Fixed Cost ($/ton)", value=200)

st.header("Variable Cost Ranges")
def get_variable_cost_input(label, default_min, default_mode, default_max):
    st.subheader(label)
    min_val = st.number_input(f"Minimum {label} ($/ton)", value=default_min)
    mode_val = st.number_input(f"Most Likely {label} ($/ton)", value=default_mode)
    max_val = st.number_input(f"Maximum {label} ($/ton)", value=default_max)
    return min_val, mode_val, max_val

resin_min, resin_mode, resin_max = get_variable_cost_input("LDPE Resin Cost", 1400, 1500, 1600)
energy_min, energy_mode, energy_max = get_variable_cost_input("Energy Cost", 100, 150, 200)
chemicals_min, chemicals_mode, chemicals_max = get_variable_cost_input("Chemicals/Additives Cost", 20, 30, 40)

st.header("Fixed Cost Ranges")
def get_fixed_cost_input(label, default_min, default_mode, default_max):
    st.subheader(label)
    min_val = st.number_input(f"Minimum {label} ($/ton)", value=default_min)
    mode_val = st.number_input(f"Most Likely {label} ($/ton)", value=default_mode)
    max_val = st.number_input(f"Maximum {label} ($/ton)", value=default_max)
    return min_val, mode_val, max_val

labor_min, labor_mode, labor_max = get_fixed_cost_input("Labor Cost", 50, 75, 100)
maintenance_min, maintenance_mode, maintenance_max = get_fixed_cost_input("Maintenance Cost", 50, 75, 100)
depreciation_min, depreciation_mode, depreciation_max = get_fixed_cost_input("Depreciation Cost", 50, 75, 100)
overhead_min, overhead_mode, overhead_max = get_fixed_cost_input("Overhead/Administration Cost", 20, 40, 60)

if st.button("Run Simulation"):
    try:
        # Generate samples using triangular distribution for selling prices
        price_sim = np.random.triangular(price_min, price_mode, price_max, n_simulations)

        # Generate samples using triangular distribution for variable costs
        resin_sim = np.random.triangular(resin_min, resin_mode, resin_max, n_simulations)
        energy_sim = np.random.triangular(energy_min, energy_mode, energy_max, n_simulations)
        chemicals_sim = np.random.triangular(chemicals_min, chemicals_mode, chemicals_max, n_simulations)

        # Generate samples using triangular distribution for fixed costs
        labor_sim = np.random.triangular(labor_min, labor_mode, labor_max, n_simulations)
        maintenance_sim = np.random.triangular(maintenance_min, maintenance_mode, maintenance_max, n_simulations)
        depreciation_sim = np.random.triangular(depreciation_min, depreciation_mode, depreciation_max, n_simulations)
        overhead_sim = np.random.triangular(overhead_min, overhead_mode, overhead_max, n_simulations)

        # Sum variable and fixed costs
        variable_cost_sim = resin_sim + energy_sim + chemicals_sim + variable_cost_base
        fixed_cost_sim = labor_sim + maintenance_sim + depreciation_sim + overhead_sim + fixed_cost_base

        total_cost_sim = variable_cost_sim + fixed_cost_sim

        # Calculate gross margin
        gross_margin = price_sim - total_cost_sim

        # Build DataFrame
        df_sim = pd.DataFrame({
            'Selling Price': price_sim,
            'Variable Cost': variable_cost_sim,
            'Fixed Cost': fixed_cost_sim,
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
        ### LDPE Pricing and Margin Simulation Report

        **Number of Simulations**: {n_simulations}

        **Selling Price Range**: ${price_min} - ${price_max} (Mode: ${price_mode})

        **Base Variable Cost**: ${variable_cost_base}/ton
        **Base Fixed Cost**: ${fixed_cost_base}/ton

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
            file_name="ldpe_pricing_simulation_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"An error occurred: {e}")
