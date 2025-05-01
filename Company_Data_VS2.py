import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="📊 Company Financial Dashboard", layout="centered")
st.title("📈 Company Financial Dashboard")

st.markdown("Enter a valid **stock ticker** (e.g., `DOW`, `LYB`, `DD`, `CE`) to retrieve key financial data.")

ticker_input = st.text_input("Enter Ticker Symbol", value="", max_chars=10)

if st.button("Get Company Data"):

    if not ticker_input:
        st.warning("⚠️ Please enter a ticker symbol.")
    else:
        try:
            ticker = yf.Ticker(ticker_input.strip().upper())
            info = ticker.info

            # Redundant error checks
            if 'shortName' not in info or info['shortName'] is None:
                st.error("❌ No valid data found. Please check the ticker symbol and try again.")
            else:
                st.success(f"✅ Data retrieved for {info.get('shortName', 'N/A')}")

                # Extract key financial metrics with fallbacks
                metrics = {
                    "Company Name": info.get("shortName", "N/A"),
                    "Industry": info.get("industry", "N/A"),
                    "Sector": info.get("sector", "N/A"),
                    "Market Cap": info.get("marketCap", "N/A"),
                    "Total Revenue": info.get("totalRevenue", "N/A"),
                    "EBITDA": info.get("ebitda", "N/A"),
                    "Trailing P/E": info.get("trailingPE", "N/A"),
                    "Forward P/E": info.get("forwardPE", "N/A"),
                    "PEG Ratio": info.get("pegRatio", "N/A"),
                    "Price to Sales": info.get("priceToSalesTrailing12Months", "N/A"),
                    "Dividend Yield": info.get("dividendYield", "N/A"),
                    "Beta": info.get("beta", "N/A"),
                    "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
                    "52 Week Low": info.get("fiftyTwoWeekLow", "N/A")
                }

                # Display as table
                df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
                st.table(df_metrics)

                # Download as CSV
                csv = df_metrics.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Financial Data as CSV",
                    data=csv,
                    file_name=f"{ticker_input.strip().upper()}_financial_data.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")
else:
    st.info("Enter a ticker symbol and click the button to fetch data.")
