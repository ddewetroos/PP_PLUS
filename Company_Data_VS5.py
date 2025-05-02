import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Company Financial Ratios", layout="centered")
st.title("📊 Company Financial Ratios Dashboard")

st.markdown("Select or enter a valid **stock ticker** (e.g., `DOW`, `LYB`, `EMN`, `CE`) to view key financial ratios. Data can be downloaded as CSV.")

# Example tickers from the chemical industry
default_tickers = ["DOW", "LYB", "EMN", "CE", "DD", "BASFY", "CLF"]

ticker_input = st.text_input("Enter Ticker Symbol", value="DOW").strip().upper()

# Button to trigger data fetch
if st.button("Fetch Financial Ratios"):
    if not ticker_input:
        st.warning("⚠️ Please enter a ticker symbol.")
    else:
        try:
            ticker = yf.Ticker(ticker_input)
            info = ticker.info

            # Validate the data returned
            if not info or 'shortName' not in info:
                st.error("❌ No valid data found for this ticker. Please check the symbol and try again.")
            else:
                st.success(f"✅ Data retrieved for: {info.get('shortName', ticker_input)}")

                # Extract financial ratios with fallback
                ratios = {
                    "Company Name": info.get("shortName", "N/A"),
                    "Sector": info.get("sector", "N/A"),
                    "Industry": info.get("industry", "N/A"),
                    "Market Cap": info.get("marketCap", "N/A"),
                    "Trailing P/E": info.get("trailingPE", "N/A"),
                    "Forward P/E": info.get("forwardPE", "N/A"),
                    "PEG Ratio": info.get("pegRatio", "N/A"),
                    "Price to Book": info.get("priceToBook", "N/A"),
                    "Price to Sales (TTM)": info.get("priceToSalesTrailing12Months", "N/A"),
                    "Enterprise Value to EBITDA": info.get("enterpriseToEbitda", "N/A"),
                    "Return on Equity (ROE)": info.get("returnOnEquity", "N/A"),
                    "Return on Assets (ROA)": info.get("returnOnAssets", "N/A"),
                    "Debt to Equity": info.get("debtToEquity", "N/A"),
                    "Profit Margin": info.get("profitMargins", "N/A"),
                    "Operating Margin": info.get("operatingMargins", "N/A"),
                    "Dividend Yield": info.get("dividendYield", "N/A"),
                    "Beta": info.get("beta", "N/A"),
                    "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                    "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
                }

                # Convert to DataFrame
                df_ratios = pd.DataFrame(ratios.items(), columns=["Metric", "Value"])

                # Display
                st.dataframe(df_ratios)

                # Offer download
                csv = df_ratios.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Ratios as CSV",
                    data=csv,
                    file_name=f"{ticker_input}_financial_ratios.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ An error occurred while fetching data: {e}")
else:
    st.info("Enter a ticker and click the button to retrieve data.")
