import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="LDPE Pricing Simulator", layout="centered")
st.title("LDPE Pricing and Margin Simulator")

st.markdown("""
This app uses **Monte Carlo simulation** to model the impact of variable **selling prices** and **cost structures** on your **gross margin**.
""")

# User inputs for number of simulations
n_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

st.header("Probability Distributions")

# Selling Price Distribution
st.subheader("Selling Price Distribution")
price_options = st.text_input("Selling Prices ($/ton) [comma-separated]", "1900, 2000, 2100, 2200, 2300, 2400")
price_probs = st.text_input("Corresponding Probabilities [comma-separated]", "0.05, 0.15, 0.3, 0.3, 0.15, 0.05")

# Cost Distribution
st.subheader("Cost Distribution")
cost_options = st.text_input("Cost Values ($/ton) [comma-separated]", "1800, 1900, 2000, 2100")
cost_probs = st.text_input("Corresponding Probabilities [comma-separated]", "0.1, 0.4, 0.4, 0.1")

if st.button("Run Simulation"):
    try:
        # Parse user inputs
        price_values = [int(p.strip()) for p in price_options.split(",")]
        price_probabilities = [float(p.strip()) for p in price_probs.split(",")]

        cost_values = [int(c.strip()) for c in cost_options.split(",")]
        cost_probabilities = [float(c.strip()) for c in cost_probs.split(",")]

        # Validate probabilities
        if not (np.isclose(sum(price_probabilities), 1.0) and np.isclose(sum(cost_probabilities), 1.0)):
            st.error("Probabilities must sum to 1.0 for both price and cost distributions.")
        else:
            # Run Monte Carlo simulation
            price_sim = np.random.choice(price_values, size=n_simulations, p=price_probabilities)
            cost_sim = np.random.choice(cost_values, size=n_simulations, p=cost_probabilities)

            gross_margin = price_sim - cost_sim

            # Build DataFrame
            df_sim = pd.DataFrame({
                'Selling Price': price_sim,
                'Cost Per Ton': cost_sim,
                'Gross Margin': gross_margin
            })

            st.subheader("Simulation Summary")
            st.write(df_sim.describe())

            st.subheader("Sample Results (First 100 Simulations)")
            st.write(df_sim.head(100))

            st.subheader("Gross Margin Distribution")
            st.bar_chart(df_sim['Gross Margin'].value_counts().sort_index())

    except Exception as e:
        st.error(f"An error occurred: {e}")
