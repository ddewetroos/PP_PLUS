import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="LDPE Pricing Simulator", layout="centered")
st.title("LDPE Pricing and Margin Simulator with Triangular Distribution")

st.markdown("""
This app uses **Monte Carlo simulation** with **Triangular Distributions** to model variable **selling prices** and **detailed cost structures** on your **gross margin**.
""")

n_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

st.header("Selling Price Range")
price_min = st.number_input("Minimum Selling Price ($/ton)", value=1900)
price_mode = st.number_input("Most Likely Selling Price ($/ton)", value=2200)
price_max = st.number_input("Maximum Selling Price ($/ton)", value=2400)

st.header("Cost Ranges for Each Component")
def get_cost_input(label, default_min, default_mode, default_max):
    st.subheader(label)
    min_val = st.number_input(f"Minimum {label} ($/ton)", value=default_min)
    mode_val = st.number_input(f"Most Likely {label} ($/ton)", value=default_mode)
    max_val = st.number_input(f"Maximum {label} ($/ton)", value=default_max)
    return min_val, mode_val, max_val

resin_min, resin_mode, resin_max = get_cost_input("LDPE Resin Cost", 1400, 1500, 1600)
energy_min, energy_mode, energy_max = get_cost_input("Energy Cost", 100, 150, 200)
labor_min, labor_mode, labor_max = get_cost_input("Labor Cost", 100, 120, 140)
packaging_min, packaging_mode, packaging_max = get_cost_input("Packaging Cost", 40, 50, 60)
logistics_min, logistics_mode, logistics_max = get_cost_input("Logistics Cost", 60, 80, 100)
overhead_min, overhead_mode, overhead_max = get_cost_input("Overhead & Depreciation", 80, 100, 120)

if st.button("Run Simulation"):
    try:
        # Generate samples using triangular distribution
        price_sim = np.random.triangular(price_min, price_mode, price_max, n_simulations)

        resin_sim = np.random.triangular(resin_min, resin_mode, resin_max, n_simulations)
        energy_sim = np.random.triangular(energy_min, energy_mode, energy_max, n_simulations)
        labor_sim = np.random.triangular(labor_min, labor_mode, labor_max, n_simulations)
        packaging_sim = np.random.triangular(packaging_min, packaging_mode, packaging_max, n_simulations)
        logistics_sim = np.random.triangular(logistics_min, logistics_mode, logistics_max, n_simulations)
        overhead_sim = np.random.triangular(overhead_min, overhead_mode, overhead_max, n_simulations)

        # Total cost per simulation
        cost_sim = resin_sim + energy_sim + labor_sim + packaging_sim + logistics_sim + overhead_sim

        # Gross margin
        gross_margin = price_sim - cost_sim

        # Build DataFrame
        df_sim = pd.DataFrame({
            'Selling Price': price_sim,
            'Total Cost': cost_sim,
            'Gross Margin': gross_margin
        })

        st.subheader("Simulation Summary")
        st.write(df_sim.describe())

        st.subheader("Sample Results (First 100 Simulations)")
        st.write(df_sim.head(100))

        st.subheader("Gross Margin Distribution")
        st.bar_chart(df_sim['Gross Margin'].round().value_counts().sort_index())

    except Exception as e:
        st.error(f"An error occurred: {e}")
