import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Cost Estimation Toolkit", layout="centered")

# Sidebar for method selection
method = st.sidebar.selectbox("Select Cost Estimation Method", [
    "Cost Index Method",
    "Power Sizing Method",
    "Factor Method",
    "Equipment Factored Estimation",
    "Price Setting Model"
])

st.title(f"{method}")

if method == "Cost Index Method":
    st.subheader("Adjust cost using historical and current cost indices")
    st.markdown("""
    **Formula:**

    \( \text{Cost}_{\text{new}} = \text{Cost}_{\text{old}} \times \left(\frac{\text{Index}_{\text{new}}}{\text{Index}_{\text{old}}}\right) \)

    **Tip:** Use CEPCI (Chemical Engineering Plant Cost Index) or another reliable industry index.
    """)
    cost_old = st.number_input("Past Known Cost", min_value=0.0)
    index_old = st.number_input("Old Cost Index", min_value=0.01)
    index_new = st.number_input("New Cost Index", min_value=0.01)

    if st.button("Estimate Updated Cost"):
        cost_new = cost_old * (index_new / index_old)
        st.success(f"Estimated Updated Cost: {cost_new:,.2f}")

        df = pd.DataFrame(
            {"Cost Old": [cost_old], "Index Old": [index_old], "Index New": [index_new], "Cost New": [cost_new]})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result as CSV", data=csv, file_name="cost_index_estimate.csv", mime="text/csv")

elif method == "Power Sizing Method":
    st.subheader("Scale cost based on capacity")
    st.markdown("""
    **Formula:**

    \( \text{Cost}_B = \text{Cost}_A \times \left(\frac{\text{Size}_B}{\text{Size}_A}\right)^x \)

    **Tip:** Common exponent values (x):
    - Civil: 0.5 to 0.7
    - Chemical process equipment: 0.6 to 0.8
    """)
    cost_A = st.number_input("Known Cost (Cost A)", min_value=0.0)
    size_A = st.number_input("Size A (Original Capacity)", min_value=0.01)
    size_B = st.number_input("Size B (New Capacity)", min_value=0.01)
    exponent = st.slider("Scaling Exponent (typically 0.5 to 0.8)", 0.0, 1.5, 0.6)

    if st.button("Estimate Scaled Cost"):
        cost_B = cost_A * (size_B / size_A) ** exponent
        st.success(f"Estimated Scaled Cost: {cost_B:,.2f}")

        df = pd.DataFrame(
            {"Cost A": [cost_A], "Size A": [size_A], "Size B": [size_B], "Exponent": [exponent], "Cost B": [cost_B]})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result as CSV", data=csv, file_name="power_sizing_estimate.csv", mime="text/csv")

elif method == "Factor Method":
    st.subheader("Estimate total cost from base using a multiplier")
    st.markdown("""
    **Formula:**

    \( \text{Total Cost} = \text{Base Cost} \times \text{Factor} \)

    **Typical Lang Factors:**
    - Solid-fluid processing plant: 4.74
    - Fluid processing plant: 3.63
    - Solid processing plant: 3.10
    """)
    base_cost = st.number_input("Base Cost (e.g., Equipment Cost)", min_value=0.0)
    factor = st.number_input("Multiplier (Lang Factor, etc.)", min_value=0.0)

    if st.button("Estimate Total Cost"):
        total_cost = base_cost * factor
        st.success(f"Estimated Total Cost: {total_cost:,.2f}")

        df = pd.DataFrame({"Base Cost": [base_cost], "Factor": [factor], "Total Cost": [total_cost]})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result as CSV", data=csv, file_name="factor_method_estimate.csv", mime="text/csv")

elif method == "Equipment Factored Estimation":
    st.subheader("Sum up cost from equipment and factors")
    st.markdown("""
    **Approach:**

    Estimate total installed cost by applying common installation and service factors to equipment cost.

    **Typical Values:**
    - Installation: 1.0 to 1.5
    - Piping: 0.3 to 0.6
    - Electrical: 0.2 to 0.4
    - Instrumentation: 0.15 to 0.3
    """)
    equipment_cost = st.number_input("Base Equipment Cost", min_value=0.0)
    install_factor = st.number_input("Installation Factor", value=1.2)
    piping_factor = st.number_input("Piping Factor", value=0.4)
    electrical_factor = st.number_input("Electrical Factor", value=0.3)
    instrumentation_factor = st.number_input("Instrumentation Factor", value=0.2)

    if st.button("Estimate Total Installed Cost"):
        total_installed = equipment_cost * (
                1 + install_factor + piping_factor + electrical_factor + instrumentation_factor
        )
        st.success(f"Estimated Total Installed Cost: {total_installed:,.2f}")

        df = pd.DataFrame({
            "Equipment Cost": [equipment_cost],
            "Install Factor": [install_factor],
            "Piping Factor": [piping_factor],
            "Electrical Factor": [electrical_factor],
            "Instrumentation Factor": [instrumentation_factor],
            "Total Installed Cost": [total_installed]
        })
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result as CSV", data=csv, file_name="equipment_factored_estimate.csv",
                           mime="text/csv")

elif method == "Price Setting Model":
    st.subheader("Set selling price based on cost and desired margin")
    st.markdown("""
    **Formula:**

    \( \text{Selling Price} = \text{Unit Cost} \times (1 + \text{Markup}) \)

    \( \text{Profit} = \text{Selling Price} - \text{Unit Cost} \)

    **Tip:** Include both fixed and variable costs to estimate accurate unit cost.
    """)
    unit_cost = st.number_input("Estimated Unit Cost", min_value=0.0)
    markup = st.slider("Desired Markup (%)", 0, 200, 30)

    if st.button("Calculate Selling Price"):
        selling_price = unit_cost * (1 + markup / 100)
        profit = selling_price - unit_cost
        st.success(f"Recommended Selling Price: {selling_price:,.2f}")
        st.info(f"Estimated Profit per Unit: {profit:,.2f}")

        df = pd.DataFrame(
            {"Unit Cost": [unit_cost], "Markup %": [markup], "Selling Price": [selling_price], "Profit": [profit]})
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Pricing Model CSV", data=csv, file_name="price_setting_model.csv", mime="text/csv")
