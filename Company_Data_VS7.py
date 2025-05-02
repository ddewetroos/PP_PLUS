import streamlit as st
import pandas as pd
import requests

# --- App Config ---
st.set_page_config(page_title="📈 Company Ratios via Alpha Vantage", layout="wide")
st.title("📊 Company Financial Ratios")

# --- Sidebar for API Key ---
st.sidebar.header("🔐 API Key")
#api_key = st.sidebar.text_input("Enter your Alpha Vantage API Key:", type="password")
api_key = "sk-proj-2ZeD16xiuComVVuzpBI1T3BlbkFJWu5iXOVWvgozKEUtJ40f"

# --- User Ticker Input ---
ticker = st.text_input("Enter stock ticker symbol (e.g., DOW):").strip().upper()

if ticker and api_key:
    # Construct API request
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"

    try:
        response = requests.get(url)
        data = response.json()

        if "Symbol" not in data:
            st.error("❌ No data found. Please check the ticker symbol and API key.")
        else:
            # Extract ratios
            ratios = {
                "Company Name": data.get("Name", "N/A"),
                "Sector": data.get("Sector", "N/A"),
                "Industry": data.get("Industry", "N/A"),
                "Market Cap": data.get("MarketCapitalization", "N/A"),
                "Trailing P/E": data.get("PERatio", "N/A"),
                "Forward P/E": data.get("ForwardPE", "N/A"),
                "PEG Ratio": data.get("PEGRatio", "N/A"),
                "Price to Book": data.get("PriceToBookRatio", "N/A"),
                "EV/EBITDA": data.get("EVToEBITDA", "N/A"),
                "ROE": data.get("ReturnOnEquityTTM", "N/A"),
                "ROA": data.get("ReturnOnAssetsTTM", "N/A"),
                "Debt to Equity": data.get("DebtEquityRatio", "N/A"),
                "Profit Margin": data.get("ProfitMargin", "N/A"),
                "Operating Margin": data.get("OperatingMarginTTM", "N/A"),
                "Dividend Yield": data.get("DividendYield", "N/A"),
                "Beta": data.get("Beta", "N/A"),
                "52 Week High": data.get("52WeekHigh", "N/A"),
                "52 Week Low": data.get("52WeekLow", "N/A")
            }

            df_ratios = pd.DataFrame(ratios.items(), columns=["Metric", "Value"])
            st.subheader(f"📈 Financial Ratios for {ticker}")
            st.dataframe(df_ratios, use_container_width=True)

            # CSV download
            csv = df_ratios.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download as CSV",
                data=csv,
                file_name=f"{ticker}_ratios_alphavantage.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")
else:
    st.info("ℹ️ Enter both an API key and a ticker symbol to begin.")
