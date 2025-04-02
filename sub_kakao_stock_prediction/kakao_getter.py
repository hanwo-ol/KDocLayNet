import yfinance as yf
import pandas as pd

# Define the ticker symbol for Kakao on the KOSPI
ticker_symbol = "035720.KS"

# Define the date range based on your news data
# Start date is inclusive, end date is exclusive
start_date = "2023-10-24"
end_date = "2025-04-03" # Set to the day *after* your last news date

print(f"Fetching stock data for Kakao ({ticker_symbol})...")
print(f"Date Range: {start_date} to {end_date} (exclusive)")

try:
    # Download the historical data
    kakao_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    # Check if data was downloaded
    if kakao_data.empty:
        print(f"No data found for ticker {ticker_symbol} in the specified date range.")
    else:
        print("\nData downloaded successfully!")

        # Display the first few rows
        print("\nFirst 5 rows of the data:")
        print(kakao_data.head())

        # Display the last few rows to confirm the date range
        print("\nLast 5 rows of the data:")
        print(kakao_data.tail())

        # Optional: Save the data to a CSV file
        csv_filename = f"kakao_stock_data_{start_date}_to_{pd.to_datetime(end_date) - pd.Timedelta(days=1):%Y-%m-%d}.csv"
        kakao_data.to_csv(csv_filename)
        print(f"\nData saved to {csv_filename}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check the ticker symbol and your internet connection.")
