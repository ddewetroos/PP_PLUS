import streamlit as st

st.set_page_config(page_title="Cost Estimation Toolkit", layout="centered")

# Sidebar for method selection
method = st.sidebar.selectbox("Select Cost Estimation Method", [
    "Cost Index Method",
    "Power Sizing Method",
    "Factor Method",
    "Equipment Factored Estimation"
])

st.title(f"{method}")

if method == "Cost Index Method":
    st.subheader("Adjust cost using historical and current cost indices")
    cost_old = st.number_input("Past Known Cost", min_value=0.0)
    index_old = st.number_input("Old Cost Index", min_value=0.01)
    index_new = st.number_input("New Cost Index", min_value=0.01)

    if st.button("Estimate Updated Cost"):
        cost_new = cost_old * (index_new / index_old)
        st.success(f"Estimated Updated Cost: {cost_new:,.2f}")

elif method == "Power Sizing Method":
    st.subheader("Scale cost based on capacity")
    cost_A = st.number_input("Known Cost (Cost A)", min_value=0.0)
    size_A = st.number_input("Size A (Original Capacity)", min_value=0.01)
    size_B = st.number_input("Size B (New Capacity)", min_value=0.01)
    exponent = st.slider("Scaling Exponent (typically 0.5 to 0.8)", 0.0, 1.5, 0.6)

    if st.button("Estimate Scaled Cost"):
        cost_B = cost_A * (size_B / size_A) ** exponent
        st.success(f"Estimated Scaled Cost: {cost_B:,.2f}")

elif method == "Factor Method":
    st.subheader("Estimate total cost from base using a multiplier")
    base_cost = st.number_input("Base Cost (e.g., Equipment Cost)", min_value=0.0)
    factor = st.number_input("Multiplier (Lang Factor, etc.)", min_value=0.0)

    if st.button("Estimate Total Cost"):
        total_cost = base_cost * factor
        st.success(f"Estimated Total Cost: {total_cost:,.2f}")

elif method == "Equipment Factored Estimation":
    st.subheader("Sum up cost from equipment and factors")
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
