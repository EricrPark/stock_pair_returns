import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def adjust_to_next_trading_day(data, date):
    max_date = data.index.max()
    while date.strftime("%Y-%m-%d") not in data.index:
        date += timedelta(days=1)
        if date > max_date:
            return max_date
    return date


def get_stock_changes(ticker, start_date):
    today = datetime.now()
    data = get_stock_data(ticker, start_date, today)
    start_date = adjust_to_next_trading_day(data, start_date)

    def calculate_change(start_date, end_date):
        end_date = adjust_to_next_trading_day(data, end_date)
        start_price = data.loc[start_date.strftime("%Y-%m-%d")]["Close"]
        end_price = data.loc[end_date.strftime("%Y-%m-%d")]["Close"]
        return ((end_price - start_price) / start_price) * 100

    one_month_change = calculate_change(start_date, start_date + timedelta(days=30))
    three_month_change = calculate_change(start_date, start_date + timedelta(days=90))

    max_end_date = data.index.max()
    end_date_one_year = start_date + timedelta(days=365)
    if end_date_one_year > max_end_date:
        one_year_change = calculate_change(start_date, max_end_date)
    else:
        one_year_change = calculate_change(start_date, end_date_one_year)

    cagr = ((1 + one_year_change / 100) ** (1 / 1) - 1) * 100

    return {
        "One Month": one_month_change,
        "Three Months": three_month_change,
        "One Year": one_year_change,
        "CAGR": cagr
    }


def calculate_portfolio_volatility(pairs, start_date):
    today = datetime.now()
    combined_returns = []

    for long_ticker, short_ticker in pairs:
        long_data = get_stock_data(long_ticker, start_date, today)
        short_data = get_stock_data(short_ticker, start_date, today)

        long_returns = long_data['Close'].pct_change().dropna()
        short_returns = short_data['Close'].pct_change().dropna()

        pair_returns = long_returns - short_returns
        combined_returns.append(pair_returns)

    combined_returns_df = pd.concat(combined_returns, axis=1).dropna()
    portfolio_returns = combined_returns_df.mean(axis=1)

    portfolio_volatility = portfolio_returns.std() * np.sqrt(252) * 100
    return portfolio_volatility, portfolio_returns


def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() * 100
    return max_drawdown


def calculate_sharpe_ratio(returns, risk_free_rate):
    mean_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (mean_return - risk_free_rate) / volatility
    return sharpe_ratio


def get_risk_free_rate(start_date):
    treasury_data = yf.Ticker("^TNX").history(start=start_date)
    treasury_data = treasury_data.resample('D').ffill().dropna()
    average_yield = treasury_data['Close'].mean() / 100
    return average_yield


def main():
    st.title("Stock Pair Returns")

    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = 0

    with st.form("pair_form"):
        num_pairs = st.number_input("Number of Pairs", min_value=1, value=1, step=1)
        pairs = []

        for i in range(num_pairs):
            col1, col2 = st.columns(2)
            with col1:
                long_ticker = st.text_input(f"Long Ticker {i + 1}")
            with col2:
                short_ticker = st.text_input(f"Short Ticker {i + 1}")
            pairs.append((long_ticker, short_ticker))

        start_date = st.date_input("Start Date", value=datetime(2023, 7, 17))
        submit_button = st.form_submit_button(label="Calculate Returns")

    if submit_button:
        start_date = start_date.strftime("%Y-%m-%d")
        adjusted_start_date = datetime.strptime(start_date, "%Y-%m-%d")

        risk_free_rate = get_risk_free_rate(adjusted_start_date)
        st.session_state.risk_free_rate = risk_free_rate

        results = []
        total_returns = {"One Month": 0, "Three Months": 0, "One Year": 0, "CAGR": 0}
        all_combined_returns = []

        for long_ticker, short_ticker in pairs:
            long_results = get_stock_changes(long_ticker, adjusted_start_date)
            short_results = get_stock_changes(short_ticker, adjusted_start_date)

            difference = {
                period: long_results[period] - short_results[period]
                for period in long_results
            }

            formatted_difference = {
                period: f"{difference[period]:.2f}%" for period in difference
            }

            pair_volatility, pair_returns = calculate_portfolio_volatility([(long_ticker, short_ticker)],
                                                                           adjusted_start_date)
            pair_max_drawdown = calculate_max_drawdown(pair_returns)
            pair_sharpe_ratio = calculate_sharpe_ratio(pair_returns, risk_free_rate)

            all_combined_returns.append(pair_returns)

            results.append({
                "Long Ticker": long_ticker,
                "Short Ticker": short_ticker,
                "Difference (One Month)": formatted_difference["One Month"],
                "Difference (Three Months)": formatted_difference["Three Months"],
                "Difference (One Year)": formatted_difference["One Year"],
                "Difference (CAGR)": formatted_difference["CAGR"],
                "Pair Volatility": f"{pair_volatility:.2f}%",
                "Max Drawdown": f"{pair_max_drawdown:.2f}%",
                "Sharpe Ratio": f"{pair_sharpe_ratio:.2f}"
            })

            # Aggregate total returns
            num_pairs = len(pairs)
            for period in total_returns:
                total_returns[period] += difference[period] / num_pairs

        # Calculate total portfolio volatility and returns
        total_volatility, total_portfolio_returns = calculate_portfolio_volatility(pairs, adjusted_start_date)
        total_max_drawdown = calculate_max_drawdown(total_portfolio_returns)
        total_sharpe_ratio = calculate_sharpe_ratio(total_portfolio_returns, risk_free_rate)

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Calculate total portfolio returns
        total_returns_formatted = {period: f"{total_returns[period]:.2f}%" for period in total_returns}
        total_volatility_formatted = f"{total_volatility:.2f}%"
        total_max_drawdown_formatted = f"{total_max_drawdown:.2f}%"
        total_sharpe_ratio_formatted = f"{total_sharpe_ratio:.2f}"

        # Append total portfolio results
        total_row = pd.Series({
            "Long Ticker": "Total Portfolio",
            "Short Ticker": "",
            "Difference (One Month)": total_returns_formatted["One Month"],
            "Difference (Three Months)": total_returns_formatted["Three Months"],
            "Difference (One Year)": total_returns_formatted["One Year"],
            "Difference (CAGR)": total_returns_formatted["CAGR"],
            "Pair Volatility": total_volatility_formatted,
            "Max Drawdown": total_max_drawdown_formatted,
            "Sharpe Ratio": total_sharpe_ratio_formatted
        })

        results_df = pd.concat([results_df, total_row.to_frame().T], ignore_index=True)

        # Convert DataFrame to HTML
        results_html = results_df.to_html(index=False)

        # Save HTML to file
        with open("stock_returns.html", "w") as file:
            file.write(results_html)

        st.success("Results saved to stock_returns.html")
        st.write(results_df)

        # Advanced details to show calculated risk-free rate
        with st.expander("Advanced Details"):
            st.write(f"Calculated Risk-Free Rate: {st.session_state.risk_free_rate * 100:.2f}%")


if __name__ == "__main__":
    main()
