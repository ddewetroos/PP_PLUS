import streamlit as st
import requests
import pandas as pd

# Set up page
st.set_page_config(page_title="FMP Company Dashboard", layout="centered")
st.title("📊 Company Financial Dashboard (FMP API)")

# API key input
#api_key = st.secrets["91ef29b8b78a6b3fac0d21ac595a42a3"] if "91ef29b8b78a6b3fac0d21ac595a42a3" in st.secrets else st.text_input("Enter your FMP API Key", type="password")
api_key = '91ef29b8b78a6b3fac0d21ac595a42a3'

# Ticker input
ticker_input = st.text_input("Enter Ticker Symbol", max_chars=10)

# Button
if st.button("Get Financial Data"):

    if not api_key:
        st.error("❌ Please enter a valid API key.")
    elif not ticker_input:
        st.warning("⚠️ Please enter a ticker symbol.")
    else:
        ticker = ticker_input.strip().upper()

        try:
            # Request data
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
            response = requests.get(url)
            profile_data = response.json()

            if not profile_data or isinstance(profile_data, dict) and profile_data.get("Error Message"):
                st.error("❌ No data found. Check the ticker symbol or your API key.")
            else:
                profile = profile_data[0]
                name = profile.get("companyName", "N/A")

                st.success(f"✅ Data retrieved for {name}")

                # Get key metrics
                url_key_metrics = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={api_key}"
                metrics_resp = requests.get(url_key_metrics).json()

                if not metrics_resp:
                    st.warning("⚠️ No key financial metrics available.")
                else:
                    metrics = metrics_resp[0]

                    selected_metrics = {
                        "Company Name": name,
                        "Market Cap": profile.get("mktCap", "N/A"),
                        "Revenue TTM": metrics.get("revenueTTM", "N/A"),
                        "EBITDA": metrics.get("ebitdaTTM", "N/A"),
                        "Trailing P/E": metrics.get("peTTM", "N/A"),
                        "Forward P/E": metrics.get("forwardPE", "N/A"),
                        "P/S Ratio": metrics.get("priceToSalesTTM", "N/A"),
                        "EV/EBITDA": metrics.get("evToEbitdaTTM", "N/A"),
                        "Debt/Equity": metrics.get("debtEquityTTM", "N/A"),
                        "Return on Equity": metrics.get("roeTTM", "N/A"),
                        "Dividend Yield": profile.get("lastDiv", "N/A")
                    }

                    df_metrics = pd.DataFrame(selected_metrics.items(), columns=["Metric", "Value"])
                    st.table(df_metrics)

                    # Download option
                    csv = df_metrics.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download as CSV", csv, f"{ticker}_financials.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
else:
    st.info("Enter your FMP API key and ticker symbol to begin.")
