import streamlit as st
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from functools import reduce


from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
import datetime


# Get stock data from Yahoo Finance API
def get_data(tickers):
    stock_data = yf.download(tickers=tickers, period="max")
    stock_data = stock_data.dropna()
    return stock_data


def get_stock(ticker, start, end):
    print(ticker)
    data = yf.download(ticker, start=start, end=end)
    data[ticker] = data['Close']
    data = data[[ticker]]
    # print(data.head())
    return data


def combine_stocks(tickers, start, end):
    data_frames = []
    for i in tickers:
        data_frames.append(get_stock(i, start, end))

    df_merged = reduce(lambda left, right: pd.merge(
        left, right, on=['Date'], how='outer'), data_frames)
    print(df_merged.head())
    return df_merged

# Calculate expected annual returns and risk for each stock


def calculate_returns_and_risks(stock_data):
    mu = expected_returns.mean_historical_return(stock_data)
    S = risk_models.sample_cov(stock_data)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    # expected_returns.returns_from_prices
    cleaned_weights = ef.clean_weights()
    rets = expected_returns.returns_from_prices(cleaned_weights, mu)
    risks = pd.Series([risk_models.portfolio_variance(cleaned_weights, S)])
    return cleaned_weights, rets, risks


# Display results for each stock
def display_results(cleaned_weights, rets, risks):
    st.write("**Expected Annual Returns and Risk for Each Stock**")
    for stock, weight in cleaned_weights.items():
        st.write("- {}:".format(stock))
        st.write("  - Expected Annual Return: {:.2%}".format(rets[stock]))
        st.write("  - Risk (Variance): {:.2%}".format(risks[0]))


def main():

    # Dow Jones stock symbols and their names
    DOW_JONES_STOCKS = {
        'AAPL': 'Apple Inc.',
        'AMGN': 'Amgen Inc.',
        'AXP': 'American Express Co.',
        'BA': 'Boeing Co.',
        'CAT': 'Caterpillar Inc.',
        'CRM': 'Salesforce.com Inc.',
        'CSCO': 'Cisco Systems Inc.',
        'CVX': 'Chevron Corp.',
        'DIS': 'Walt Disney Co.',
        'DOW': 'Dow Inc.',
        'GS': 'Goldman Sachs Group Inc.',
        'HD': 'Home Depot Inc.',
        'HON': 'Honeywell International Inc.',
        'IBM': 'International Business Machines Corp.',
        'INTC': 'Intel Corp.',
        'JNJ': 'Johnson & Johnson',
        'JPM': 'JPMorgan Chase & Co.',
        'KO': 'Coca-Cola Co.',
        'MCD': 'McDonald\'s Corp.',
        'MMM': '3M Co.',
        'MRK': 'Merck & Co. Inc.',
        'MSFT': 'Microsoft Corp.',
        'NKE': 'Nike Inc.',
        'PG': 'Procter & Gamble Co.',
        'TRV': 'Travelers Companies Inc.',
        'UNH': 'UnitedHealth Group Inc.',
        'V': 'Visa Inc.',
        'VZ': 'Verizon Communications Inc.',
        'WBA': 'Walgreens Boots Alliance Inc.',
        'WMT': 'Walmart Inc.',
        'XOM': 'Exxon Mobil Corp.'
    }

    # Streamlit app title and layout
    st.set_page_config(page_title="Dashboard", layout="wide")

    # Title
    # st.title("Dashboard")

    # sidebar
    st.sidebar.title("Select Task")

    option = ["Dashboard", "Predictions", "Portfolio Optimization"]
    choice = st.sidebar.selectbox("Select Task", option)

    if choice == "Dashboard":
        st.subheader("Dashboard")
        # Dropdown menu
        selected_stock = st.selectbox("Select a Dow Jones stock", list(
            DOW_JONES_STOCKS.keys()), format_func=lambda x: f"{x} ({DOW_JONES_STOCKS[x]})")

        # Get the stock data for the past 1 year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        data = yf.download(selected_stock, start=start_date, end=end_date)

        # Plot the stock chart
        fig = px.line(data, x=data.index, y="Close",
                      title=f"{selected_stock} ({DOW_JONES_STOCKS[selected_stock]}) - Past 1 Year")
        st.plotly_chart(fig, use_container_width=True)

    elif choice == "Portfolio Optimization":

        # ... (existing code)
        st.subheader("Portfolio Optimization")
        # Portfolio Optimization Module
        st.title("Portfolio Optimization")

        start = st.date_input("enter start date", datetime.date(2021, 5, 17))
        end = st.date_input("enter end date", datetime.date(2022, 5, 17))

        # # multiselect
        selected_stocks = st.multiselect(
            "Select your stocks", DOW_JONES_STOCKS.keys())
        st.write(selected_stocks)

        if st.button("Optimize Portfolio"):
            portfolio = combine_stocks(selected_stocks, start, end)
            # st.dataframe(portfolio)

            mu = mean_historical_return(portfolio)
            S = CovarianceShrinkage(portfolio).ledoit_wolf()
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()

            ef.portfolio_performance(verbose=True) 

                        
            # latest_prices = get_latest_prices(portfolio)

            # da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=100000)

            # allocation, leftover = da.greedy_portfolio()
            # st.write(print("Discrete allocation:", allocation)
            # print("Funds remaining: ${:.2f}".format(leftover))


        # st.write('You selected:', selected_stocks)_JONES_STOCKS[x]})")
        # Time duration input
        # duration = st.slider("Time duration in years", 1, 10, 1)

        # # Amount input
        # investment_amount = st.number_input(
        #     "Investment amount", min_value=1000, value=10000, step=1000)

        # # multiselect
        # selected_stocks = st.multiselect(
        #     "Select your stocks", DOW_JONES_STOCKS.keys())
        # # st.write('You selected:', selected_stocks)

        # # Select stocks and amount of money to invest
        # stocks = st.multiselect(
        #     "Select Stocks", ["AAPL", "GOOG", "AMZN", "FB", "TSLA"])
        # money = st.number_input(
        #     "Enter Amount of Money to Invest", min_value=1, step=1)

        # # Get stock data and calculate expected returns and risks
        # if stocks:
        #     stock_data = get_data(stocks)
        #     cleaned_weights, rets, risks = calculate_returns_and_risks(
        #         stock_data)
        #     # Display results
        #     display_results(cleaned_weights, rets, risks)

        #     # Calculate and display recommended allocation of money for each stock
        #     latest_prices = get_latest_prices(stock_data)
        #     da = DiscreteAllocation(
        #         cleaned_weights, latest_prices, total_portfolio_value=money)
        #     allocation, leftover = da.lp_portfolio()
        #     st.write("**Recommended Allocation of Money for Each Stock**")
        #     for stock, shares in allocation.items():
        #         st.write("- {}:".format(stock))
        #         st.write("  - Shares: {}".format(shares))
        #         st.write(
        #             "  - Cost: ${:.2f}".format(shares * latest_prices[stock]))
        #     st.write("**Leftover Money**")
        #     st.write("- ${:.2f}".format(leftover))

        # Portfolio optimization button
        # if st.button("Optimize Portfolio"):

        # # Fetch stock data based on the user's time duration input
        # end_date = datetime.now()
        # start_date = end_date - timedelta(days=duration * 365)
        # stock_data = yf.download(
        #     list(DOW_JONES_STOCKS.keys()), start=start_date, end=end_date)["Adj Close"]
        # # Calculate expected returns and the covariance matrix of the stocks
        # mu = expected_returns.mean_historical_return(stock_data)
        # S = risk_models.sample_cov(stock_data)
        # # Optimize the portfolio using the Efficient Frontier
        # ef = EfficientFrontier(mu, S)
        # weights = ef.max_sharpe()
        # cleaned_weights = ef.clean_weights()
        # # Display the optimized portfolio allocation
        # st.subheader("Optimized Portfolio Allocation")
        # for stock, weight in cleaned_weights.items():
        #     st.write(
        #         f"{stock} ({DOW_JONES_STOCKS[stock]}): {weight * 100:.2f}%")
        # st.title("Stock Diversification App")
if __name__ == "__main__":
    main()
