import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st


def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)

    # Ensure the index is a DatetimeIndex and remove timezone if needed
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    if data.index.tzinfo is not None:
        data.index = data.index.tz_localize(None)  # Remove timezone information if present

    return data


def get_stock_changes(ticker, start_date):
    today = datetime.now().strftime("%Y-%m-%d")
    data = get_stock_data(ticker, start_date, today)

    # Check if 'Adj Close' column is available
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

    def get_closest_date(date):
        date = pd.Timestamp(date)  # Ensure date is a Timestamp
        if date > data.index[-1]:
            return data.index[-1]
        while date not in data.index:
            date -= timedelta(days=1)
        return date

    def calculate_change(end_date):
        end_date = pd.Timestamp(end_date)  # Ensure end_date is a Timestamp
        start_price = data.loc[get_closest_date(start_date)][price_column]
        end_price = data.loc[get_closest_date(end_date)][price_column]
        return ((end_price - start_price) / start_price) * 100

    def calculate_cagr():
        start_date_ts = pd.Timestamp(start_date)  # Ensure start_date is a Timestamp
        end_date_ts = pd.Timestamp(today)  # Ensure today is a Timestamp
        start_price = data.loc[get_closest_date(start_date_ts)][price_column]
        end_price = data.loc[get_closest_date(end_date_ts)][price_column]
        num_years = (end_date_ts - start_date_ts).days / 365.25
        return ((end_price / start_price) ** (1 / num_years) - 1) * 100

    def calculate_volatility():
        data['Returns'] = data[price_column].pct_change()
        return data['Returns'].std() * np.sqrt(252) * 100  # Annualized volatility

    one_month_change = calculate_change(datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=30))
    three_month_change = calculate_change(datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=90))
    one_year_change = calculate_change(datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=365))
    cagr = calculate_cagr()
    volatility = calculate_volatility()

    return {
        "One Month": one_month_change,
        "Three Months": three_month_change,
        "One Year": one_year_change,
        "CAGR": cagr,
        "Volatility": volatility
    }


def adjust_date_to_trading_day(date):
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")

    while date.weekday() > 4:  # 5=Saturday, 6=Sunday
        date += timedelta(days=1)

    stock = yf.Ticker("AAPL")
    data = stock.history(start=date, end=date + timedelta(days=1))
    while date.strftime("%Y-%m-%d") not in data.index:
        date += timedelta(days=1)
        data = stock.history(start=date, end=date + timedelta(days=1))

    return date.strftime("%Y-%m-%d")


def calculate_pair_volatility(long_ticker, short_ticker, start_date):
    today = datetime.now().strftime("%Y-%m-%d")
    long_data = get_stock_data(long_ticker, start_date, today)
    short_data = get_stock_data(short_ticker, start_date, today)

    # Check if 'Adj Close' column is available
    price_column = 'Adj Close' if 'Adj Close' in long_data.columns else 'Close'

    long_data['Returns'] = long_data[price_column].pct_change()
    short_data['Returns'] = short_data[price_column].pct_change()

    # Merge data on the date index
    combined_data = pd.merge(long_data[['Returns']], short_data[['Returns']], left_index=True, right_index=True,
                             suffixes=('_long', '_short'))
    combined_data['Pair_Returns'] = combined_data['Returns_long'] - combined_data['Returns_short']

    # Ensure no NaN values
    combined_data.dropna(subset=['Pair_Returns'], inplace=True)

    # Calculate the annualized volatility of the pair returns
    return combined_data['Pair_Returns'].std() * np.sqrt(252) * 100  # Annualized volatility


def calculate_portfolio_volatility(pairs, start_date):
    today = datetime.now().strftime("%Y-%m-%d")

    returns = []
    for long_ticker, short_ticker in pairs:
        long_data = get_stock_data(long_ticker, start_date, today)
        short_data = get_stock_data(short_ticker, start_date, today)

        # Check if 'Adj Close' column is available
        price_column = 'Adj Close' if 'Adj Close' in long_data.columns else 'Close'

        long_data['Returns'] = long_data[price_column].pct_change()
        short_data['Returns'] = short_data[price_column].pct_change()

        # Merge data on the date index
        combined_data = pd.merge(long_data[['Returns']], short_data[['Returns']], left_index=True, right_index=True,
                                 suffixes=('_long', '_short'))
        combined_data['Pair_Returns'] = combined_data['Returns_long'] - combined_data['Returns_short']

        # Ensure no NaN values
        combined_data.dropna(subset=['Pair_Returns'], inplace=True)

        returns.append(combined_data['Pair_Returns'])

    # Concatenate all pair returns into a single DataFrame
    returns_df = pd.concat(returns, axis=1)
    returns_df.columns = [f'Pair_{i}' for i in range(len(pairs))]

    # Calculate covariance matrix
    cov_matrix = returns_df.cov()

    # Assuming equal weights for each pair
    weights = np.array([1 / len(pairs)] * len(pairs))

    # Calculate portfolio variance
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

    # Calculate portfolio volatility
    port_volatility = np.sqrt(port_variance) * np.sqrt(252) * 100  # Annualized volatility
    return port_volatility


def main():
    st.title("Stock Pair Trade Calculator")

    st.header("Input Parameters")

    # Create a form for multiple pairs
    with st.form(key='input_form'):
        num_rows = st.number_input("Number of pairs", min_value=1, value=1, step=1)
        pairs = []
        for i in range(num_rows):
            st.subheader(f"Pair {i + 1}")
            long_ticker = st.text_input(f"Enter the stock ticker for long position {i + 1}:", key=f"long_{i}")
            short_ticker = st.text_input(f"Enter the stock ticker for short position {i + 1}:", key=f"short_{i}")
            pairs.append((long_ticker, short_ticker))

        start_date = st.date_input("Enter the start date:", datetime(2023, 1, 1))
        start_date_str = start_date.strftime("%Y-%m-%d")

        submit_button = st.form_submit_button("Calculate Returns")

    if submit_button:
        adjusted_start_date = adjust_date_to_trading_day(start_date_str)

        results = []
        total_returns = {"One Month": 0, "Three Months": 0, "One Year": 0, "CAGR": 0}
        total_volatility = 0

        for long_ticker, short_ticker in pairs:
            long_results = get_stock_changes(long_ticker, adjusted_start_date)
            short_results = get_stock_changes(short_ticker, adjusted_start_date)
            pair_volatility = calculate_pair_volatility(long_ticker, short_ticker, adjusted_start_date)

            difference = {period: long_results[period] - short_results[period] for period in long_results}
            formatted_difference = {period: f"{difference[period]:.2f}%" for period in difference}

            results.append({
                "Long Ticker": long_ticker,
                "Short Ticker": short_ticker,
                "Difference (One Month)": formatted_difference["One Month"],
                "Difference (Three Months)": formatted_difference["Three Months"],
                "Difference (One Year)": formatted_difference["One Year"],
                "Difference (CAGR)": formatted_difference["CAGR"],
                "Pair Volatility": f"{pair_volatility:.2f}%"
            })

            # Aggregate total returns
            num_pairs = len(pairs)
            for period in total_returns:
                total_returns[period] += difference[period] / num_pairs

        # Calculate total portfolio volatility
        total_volatility = calculate_portfolio_volatility(pairs, adjusted_start_date)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate total portfolio returns
        total_returns_formatted = {period: f"{total_returns[period]:.2f}%" for period in total_returns}
        total_volatility_formatted = f"{total_volatility:.2f}%"

        # Append total portfolio results
        results_df = results_df.append(pd.Series({
            "Long Ticker": "Total Portfolio",
            "Short Ticker": "",
            "Difference (One Month)": total_returns_formatted["One Month"],
            "Difference (Three Months)": total_returns_formatted["Three Months"],
            "Difference (One Year)": total_returns_formatted["One Year"],
            "Difference (CAGR)": total_returns_formatted["CAGR"],
            "Pair Volatility": total_volatility_formatted
        }, name="Total"), ignore_index=True)

        # Convert DataFrame to HTML
        results_html = results_df.to_html(index=False)

        # Save HTML to file
        with open("stock_returns.html", "w") as file:
            file.write(results_html)

        st.success("Results saved to stock_returns.html")
        st.write(results_df)


if __name__ == "__main__":
    main()
