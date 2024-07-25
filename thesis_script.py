import sys, os
import numpy as np
import pandas as pd
import scipy.optimize as sc
import plotly.graph_objects as go
import yfinance as yfin
import dash
import webbrowser
from dash.dash_table import DataTable
from dash import html
from dash import dcc
from pandas_datareader import data as pdr
from statistics import stdev

# Line required for pdr.get_data_yahoo() to function properly
yfin.pdr_override()

# Number of trading days durnig a fiscal year
NB_TRADING_DAYS_PER_YEAR = 252

# Risk free rate (impacts the distance between max SR & min Variance). Set to 0% for simplicity.
riskFreeRate = 0

# Start & end dates of in-sample data
# Note: this script can only work for 1 month period!
startDate = '2022-01-01'
endDate = '2022-02-01'

# Run the script with one model portfolio at a time
################
#### FAMA-FRENCH
#Model_Portfolio = {'ENPH' : 0.2, 'TSLA' : 0.179, 'ETSY' : 0.143, 'GME' : 0.097, 'PAYC' : 0.075, 'NOW' : 0.072, 'NKTR' : 0.063, 'PTC' : 0.062, 'LW' : 0.057, 'RIG' : 0.05}
#Model_Portfolio_Label = 'Fama-French'
#ChatGPT_Weighted_Portfolio = {'ENPH' : 0.15, 'TSLA' : 0.15, 'ETSY' : 0.1, 'GME' : 0.05, 'PAYC' : 0.1, 'NOW' : 0.15, 'NKTR' : 0.05, 'PTC' : 0.1, 'LW' : 0.05, 'RIG' : 0.1}
############
#### ChatGPT (stock selection)
#Model_Portfolio = {'RIG' : 0.2, 'MRO' : 0.157, 'MUR' : 0.116, 'BFH' : 0.102, 'CCL' : 0.091, 'APA' : 0.09, 'DVN' : 0.08, 'NCLH' : 0.062, 'RCL' : 0.056, 'JWN' : 0.05}
#Model_Portfolio_Label = 'ChatGPT factor model'
#ChatGPT_Weighted_Portfolio = {'RIG' : 0.1, 'MRO' : 0.12, 'MUR' : 0.1, 'BFH' : 0.15, 'CCL' : 0.07, 'APA' : 0.12, 'DVN' : 0.13, 'NCLH' : 0.07, 'RCL' : 0.07, 'JWN' : 0.07}
###########
#### Hybrid
Model_Portfolio = {'OGN' : 0.2, 'CRL' : 0.119, 'CZR' : 0.101, 'COTY' : 0.097, 'PENN' : 0.095, 'GNRC' : 0.086, 'VNT' : 0.086, 'DXCM' : 0.083, 'MSCI' : 0.083, 'ZBRA' : 0.05}
Model_Portfolio_Label = 'Hybrid'
ChatGPT_Weighted_Portfolio = {'OGN' : 0.1, 'CRL' : 0.12, 'CZR' : 0.08, 'COTY' : 0.06, 'PENN' : 0.1, 'GNRC' : 0.12, 'VNT' : 0.1, 'DXCM' : 0.12, 'MSCI' : 0.1, 'ZBRA' : 0.1}

# Define wether the portfolios on efficient frontier with or without constraints
SCENARIO_WITH_CONSTRAINTS = False

# The minimum & maximum weight for each stock in portfolio
constraintSet = (0.05, 0.2) if SCENARIO_WITH_CONSTRAINTS else (0, 1)

# Safety check
if(abs(sum(Model_Portfolio.values()) - 1) > 0.01): sys.exit(f'Program interrupted! Sum of shares in Model_Portfolio is {round(100*sum(Model_Portfolio.values()),2)} % !')
if(abs(sum(ChatGPT_Weighted_Portfolio.values()) - 1) > 0.01): sys.exit(f'Program interrupted! Sum of shares in ChatGPT_Weighted_Portfolio is {round(100*sum(ChatGPT_Weighted_Portfolio.values()),2)} % !')

# List of stocks (tickers) in portfolios (generic & Model)
# Generic portfolio: weights are not defined and will vary giving various return & volatility values
stockList = list(Model_Portfolio.keys())

# Replace old tickers with new ones
# TSMC  >>  TWD

# Add S&P500 & Model recommended portfolios to efficientFrontierGraph() as black markers
addStockList = ['^GSPC', '^IXIC', '^DJI']
addStockDisplayNames = {'^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones'}

# It's crucial to sort tickers alphabetically for this program to work properly
Model_Portfolio = dict(sorted(Model_Portfolio.items()))
Model_Portfolio_Weights = np.array(list(Model_Portfolio.values()))
ChatGPT_Weighted_Portfolio = dict(sorted(ChatGPT_Weighted_Portfolio.items()))
ChatGPT_Portfolio_Weights = np.array(list(ChatGPT_Weighted_Portfolio.values()))
addStockList = sorted(addStockList)

# Get stocks' adjusted closing prices & calculate covariance matrix
# The adjusted closing price includes anything that would affect the stock price (stock splits, dividends...)
def getData(stocks: list, start: str, end: str):

    stockPrices = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockPrices = stockPrices['Adj Close']

    dailyStockPriceChanges = stockPrices.pct_change(fill_method=None)
    covMatrix = dailyStockPriceChanges.cov()

    return stockPrices, covMatrix

# New formula of monthly annualised return, starting from 1 month return
def stockAnnualisedReturn(stockPrices: pd.Series):
        
    initialPrice = stockPrices.iloc[0]
    finalPrice = stockPrices.iloc[-1]
    oneMonthReturn = (finalPrice - initialPrice) / initialPrice
    monthlyAnnualisedStockReturn = 100 * ((1 + oneMonthReturn)**12 - 1)

    return monthlyAnnualisedStockReturn

# Calculate non-annualised stock return over a specific period
def stockNonAnnualisedReturn(stockPrices: pd.Series):
        
    initialPrice = stockPrices.iloc[0]
    finalPrice = stockPrices.iloc[-1]
    totalReturn = (finalPrice - initialPrice) / initialPrice
    totalReturn = totalReturn*100
    return totalReturn

# Calculate annualised stock variance over a specific period
def stockAnnualisedVariance(stockPrices: pd.Series):

    # Daily percentage changes in stock price
    dailyStockPriceChanges = stockPrices.pct_change(fill_method=None)

    # Daily stock price volatility
    dailyVolatility = stdev(dailyStockPriceChanges[1:])
    annualisedDailyVolatilityInTradingDays = dailyVolatility * np.sqrt(NB_TRADING_DAYS_PER_YEAR)
    
    annualisedDailyVolatilityInTradingDays = round(annualisedDailyVolatilityInTradingDays*100,2)
    return annualisedDailyVolatilityInTradingDays

# New formula of monthly annualised return, starting from 1 month return
def portfolioAnnualisedReturn(weights: np.ndarray, stockPrices: pd.DataFrame):
    
    monthlyAnnualisedPortfolioReturn = 0
    monthlyAnnualisedReturnsPerStock_array = []
    for stock in stockPrices:

        # For each stock, calculate monthly annualised return over all period of stockPrices (1 month)
        initialPrice = stockPrices[stock].iloc[0]
        finalPrice = stockPrices[stock].iloc[-1]
        oneMonthReturn = (finalPrice - initialPrice) / initialPrice
        monthlyAnnualisedReturnsPerStock_array.append((1 + oneMonthReturn) ** 12 - 1)

    # Calculate annualised portfolio return
    for w, r in zip(weights, monthlyAnnualisedReturnsPerStock_array): monthlyAnnualisedPortfolioReturn += w * r
    return monthlyAnnualisedPortfolioReturn

# Calculate portfolio non-annualised return over a specific period
def portfolioNonAnnualisedReturn(weights: np.ndarray, stockPrices: pd.DataFrame):
    
    nonAnnualisedPortfolioReturn = 0
    nonAnnualisedReturnsPerStock_array = []
    for stock in stockPrices:

        # For each stock, calculate return over all period of stockPrices (1 month)
        initialPrice = stockPrices[stock].iloc[0]
        finalPrice = stockPrices[stock].iloc[-1]
        oneMonthReturn = (finalPrice - initialPrice) / initialPrice
        nonAnnualisedReturnsPerStock_array.append(oneMonthReturn)

    # Calculate annualised portfolio return
    for w, r in zip(weights, nonAnnualisedReturnsPerStock_array): nonAnnualisedPortfolioReturn += w * r
    return nonAnnualisedPortfolioReturn

# Calculate portfolio annualised variance over a specific period
def portfolioAnnualisedVariance(weights: np.ndarray, covMatrix: pd.DataFrame):

    # Calculate portfolio annualised daily volatility (in trading days)
    pAannualisedDailyVolatilityInTradingDays = 0
    pAannualisedDailyVolatilityInTradingDays = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(NB_TRADING_DAYS_PER_YEAR)
    
    return pAannualisedDailyVolatilityInTradingDays

# For each returnTarget, Minimise variance by altering the weights of the portfolio
def efficientOpt(stockPrices: pd.DataFrame, covMatrix: pd.DataFrame, returnTarget: float):

    numAssets = len(stockPrices.columns)
    args = (covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: portfolioAnnualisedReturn(x, stockPrices) - returnTarget},
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))
    optimalPortfolio = sc.minimize(portfolioAnnualisedVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)

    return optimalPortfolio

# Calculate negative Sharpe Ratio of the overall portfolio
def negativeSR(weights: np.ndarray, stockPrices: pd.DataFrame, covMatrix: pd.DataFrame):

    pAnnualisedReturn = portfolioAnnualisedReturn(weights, stockPrices)
    pAannualisedDailyVolatilityInTradingDays = portfolioAnnualisedVariance(weights, covMatrix)
    return - (pAnnualisedReturn - riskFreeRate)/pAannualisedDailyVolatilityInTradingDays

# Maximise Sharpe Ratio by altering the weights of the portfolio
def maximiseSR(stockPrices: pd.DataFrame, covMatrix: pd.DataFrame):

    numAssets = len(stockPrices.columns)
    args = (stockPrices, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))

    # initial guess of asset optimal weights is equal distribution of assets
    maxSRportfolio = sc.minimize(negativeSR, numAssets*[1./numAssets],
                                 args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return maxSRportfolio

# Minimise variance by altering the weights of the portfolio
def minimizeVariance(stockPrices: pd.DataFrame, covMatrix: pd.DataFrame):

    numAssets = len(stockPrices.columns)
    args = (covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraintSet for asset in range(numAssets))

    # initial guess of asset optimal weights is equal distribution of assets
    minVolPortfolio = sc.minimize(portfolioAnnualisedVariance, numAssets*[1./numAssets],
                                  args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return minVolPortfolio

# Function to remove elements from the start and end of the array if they have the same value
def removeDuplicateEnds(x_list, y_list):
    start, end = 0, len(x_list) - 1

    # Remove duplicates from the start
    while start < end and round(x_list[start], 4) == round(x_list[start + 1], 4):
        start += 1

    # Remove duplicates from the end
    while end > start and round(x_list[end], 4) == round(x_list[end - 1], 4):
        end -= 1

    # Slicing the arrays to exclude the duplicate elements
    return x_list[start:end + 1], y_list[start:end + 1]

# Return a graph ploting the min volatility, max SR and efficient frontier
def efficientFrontierGraph(
        maxSR_annualised_return: float, maxSR_std: float, 
        minVol_annualised_return: float, minVol_std: float, 
        volatilityPerTargetReturn: list, targetReturns: np.ndarray,
        GenStocks_returns: list, GenStocks_std: list,
        AddStocks_annualised_returns: list, AddStocks_std: list,
        Model_annualised_return: float, Model_std: float,
        ChatGPT_annualised_return: float, ChatGPT_std: float,
        EquWei_annualised_return: float, EquWei_std: float
    ):

    # Min. volatility portfolio
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers+text',
        x=[minVol_std],
        y=[minVol_annualised_return],
        marker=dict(color='green',size=14,line=dict(width=3, color='black')),
        text='Min Volatility',
        textposition='middle left'
    )

    # Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in volatilityPerTargetReturn],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=3, dash='dot')
    )

    data = [EF_curve, MinVol]
    constraints_or_not = '(with upper & lower constraints)' if SCENARIO_WITH_CONSTRAINTS else '(without upper & lower constraints)'
    layout = go.Layout(
        title = 'Portfolio Optimisation - Efficient Frontier, '+str(len(Model_Portfolio_Weights))+' stocks, ('+startDate+' to '+endDate+') '+constraints_or_not,
        yaxis = dict(title='Monthly Annualised Return (%)'),
        xaxis = dict(title='Annualised Daily Volatility (in trading days) (%)'),
        showlegend = False,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=1400,
        height=600)

    # Generic portfolio stocks
    for i in range(len(stockList)):
        GenStock = go.Scatter(
            name=stockList[i],
            mode='markers+text',
            x=[GenStocks_std[i]],
            y=[GenStocks_returns[i]],
            marker=dict(color='#d1c7c7',size=12,line=dict(width=2, color='#d1c7c7')),
            text=stockList[i],
            textfont=dict(size=8),
            textposition='middle right'
        )
        data.append(GenStock)

    # Equally weighted portfolio
    EquWei_marker = go.Scatter(
        name='Equally weighted',
        mode='markers+text',
        x=[EquWei_std],
        y=[EquWei_annualised_return],
        marker=dict(color='#FE9900',size=12,line=dict(width=2, color='#FE9900')),
        text='Equally weighted',
        textposition='bottom center'
    )
    data.append(EquWei_marker)

    # Additional stocks
    for i in range(len(addStockList)):

        addSingleStock = go.Scatter(
            name=addStockDisplayNames[addStockList[i]],
            mode='markers+text',
            x=[AddStocks_std[i]],
            y=[AddStocks_annualised_returns[i]],
            marker=dict(color='#807a7a',size=12,line=dict(width=2, color='#807a7a')),
            text=addStockDisplayNames[addStockList[i]],
            textposition='top center'
        )
        data.append(addSingleStock)

    # Model portfolio
    ModelPortfolioMarker = go.Scatter(
        name=Model_Portfolio_Label,
        mode='markers+text',
        x=[Model_std],
        y=[Model_annualised_return],
        marker=dict(color='#0000FF',size=12,line=dict(width=2, color='#0000FF')),
        text=Model_Portfolio_Label,
        textposition='middle right'
    )
    data.append(ModelPortfolioMarker)

    # ChatGPT weighted portfolio
    ChatGPTPortfolioMarker = go.Scatter(
        name='ChatGPT',
        mode='markers+text',
        x=[ChatGPT_std],
        y=[ChatGPT_annualised_return],
        marker=dict(color='yellow',size=12,line=dict(width=0, color='black')),
        text='ChatGPT',
        textposition='middle right'
    )
    data.append(ChatGPTPortfolioMarker)

    # Max. SR. portfolio
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio',
        mode='markers+text',
        x=[maxSR_std],
        y=[maxSR_annualised_return],
        marker=dict(color='red',size=12,line=dict(width=3, color='black')),
        text='Max SR',
        textposition='top center'
    )
    data.append(MaxSharpeRatio)

    fig = go.Figure(data=data, layout=layout)

    return fig.show()

########################################################################
# Step 0: Calculate returns & volatility
########################################################################
# Generic portfolio
stockPricesGeneric, covMatrixGeneric = getData(stockList, start=startDate, end=endDate)

# Safety check
if stockPricesGeneric.isnull().values.any(): sys.exit('Program interrupted! NaN values found in stock prices!')

# Each stock in generic portfolio
GenStocks_returns = list()
GenStocks_std = list()

for stock in stockList:
        
    monthlyAnnualisedStockReturn = stockAnnualisedReturn(stockPricesGeneric[stock])
    annualisedDailyVolatilityInTradingDays = stockAnnualisedVariance(stockPricesGeneric[stock])
    GenStocks_returns.append(monthlyAnnualisedStockReturn)
    GenStocks_std.append(annualisedDailyVolatilityInTradingDays)

# Model portfolio
Model_nn_annualised_return = portfolioNonAnnualisedReturn(Model_Portfolio_Weights, stockPricesGeneric)
Model_annualised_return = portfolioAnnualisedReturn(Model_Portfolio_Weights, stockPricesGeneric)
Model_std = portfolioAnnualisedVariance(Model_Portfolio_Weights, covMatrixGeneric)
Model_SR = round((Model_annualised_return - riskFreeRate) / Model_std, 2)
Model_nn_annualised_return, Model_annualised_return, Model_std = round(Model_nn_annualised_return*100,2), round(Model_annualised_return*100,2), round(Model_std*100,2)

# ChatGPT weighted portfolio
ChatGPT_nn_annualised_return = portfolioNonAnnualisedReturn(ChatGPT_Portfolio_Weights, stockPricesGeneric)
ChatGPT_annualised_return = portfolioAnnualisedReturn(ChatGPT_Portfolio_Weights, stockPricesGeneric)
ChatGPT_std = portfolioAnnualisedVariance(ChatGPT_Portfolio_Weights, covMatrixGeneric)
ChatGPT_SR = round((ChatGPT_annualised_return - riskFreeRate) / ChatGPT_std, 2)
ChatGPT_nn_annualised_return, ChatGPT_annualised_return, ChatGPT_std = round(ChatGPT_nn_annualised_return*100,2), round(ChatGPT_annualised_return*100,2), round(ChatGPT_std*100,2)

# Equally weighted portfolio
EquWei_weights = np.array([1/len(Model_Portfolio_Weights)] * len(Model_Portfolio_Weights))
EquWei_nn_annualised_return = portfolioNonAnnualisedReturn(EquWei_weights, stockPricesGeneric)
EquWei_annualised_return = portfolioAnnualisedReturn(EquWei_weights, stockPricesGeneric)
EquWei_std = portfolioAnnualisedVariance(EquWei_weights, covMatrixGeneric)
EquWeiSR = round((EquWei_annualised_return - riskFreeRate) / EquWei_std, 2)
EquWei_nn_annualised_return, EquWei_annualised_return, EquWei_std = round(EquWei_nn_annualised_return*100,2), round(EquWei_annualised_return*100,2), round(EquWei_std*100,2)

# Additional stocks
stockPricesAddStocks = pdr.get_data_yahoo(addStockList, start=startDate, end=endDate)
stockPricesAddStocks = stockPricesAddStocks['Adj Close']

AddStocks_annualised_returns = list()
AddStocks_nn_annualised_returns = list()
AddStocks_std = list()
AddStocks_SR = list()

for stock in addStockList:
    
    nonAnnualisedReturn = stockNonAnnualisedReturn(stockPricesAddStocks[stock])
    annualisedReturn = stockAnnualisedReturn(stockPricesAddStocks[stock])
    annualisedDailyVolatilityInTradingDays = stockAnnualisedVariance(stockPricesAddStocks[stock])
    stock_SR = round((annualisedReturn - riskFreeRate) / annualisedDailyVolatilityInTradingDays, 2)
    
    AddStocks_nn_annualised_returns.append(nonAnnualisedReturn)
    AddStocks_annualised_returns.append(annualisedReturn)
    AddStocks_std.append(annualisedDailyVolatilityInTradingDays)
    AddStocks_SR.append(stock_SR)

########################################################################
# Step 1: Max. Sharpe ratio portfolio
########################################################################
maxSRportfolio = maximiseSR(stockPricesGeneric, covMatrixGeneric)

# Interrupt the program if the optimisation has failed
if not maxSRportfolio.success: sys.exit('Program interrupted! Failed to compute the maximum Sharpe ratio portfolio!')

# Weights of max. SR portfolio
maxSR_weights = np.array(maxSRportfolio.x)

# Composition of max. SR portfolio
maxSRportfolio_df = pd.DataFrame(data=[maxSRportfolio.x], columns=covMatrixGeneric.columns).T
maxSRportfolio_df = maxSRportfolio_df.sort_values(0, ascending=False)

# Calculate return, volatility & SR. of max. Sharpe ratio portfolio
maxSR_nn_annualised_return = portfolioNonAnnualisedReturn(maxSR_weights, stockPricesGeneric)
maxSR_annualised_return = portfolioAnnualisedReturn(maxSR_weights, stockPricesGeneric)
maxSR_std = portfolioAnnualisedVariance(maxSR_weights, covMatrixGeneric)
maxSRportfolioSR = round((maxSR_annualised_return - riskFreeRate) / maxSR_std, 2)
maxSR_nn_annualised_return, maxSR_annualised_return, maxSR_std = round(maxSR_nn_annualised_return*100,2), round(maxSR_annualised_return*100,2), round(maxSR_std*100,2)

# Override Max SR Portfolio in efficient frontier if needed (sometimes the Max SR falls outside EF)
if OVERRIDE_MAX_SR_IN_EF:
    maxSR_std = MAX_SR_STD_OVERRIDE
    maxSR_annualised_return = MAX_SR_ANN_RETURN_OVERRIDE

########################################################################
# Step 2: Min. volatility portfolio
########################################################################
minVolPortfolio = minimizeVariance(stockPricesGeneric, covMatrixGeneric)

# Interrupt the program if the optimisation has failed
if not minVolPortfolio.success: sys.exit('Program interrupted! Failed to compute the minimum volatility portfolio!')

# Weights of min. volatility portfolio
minVolPortfolio_weights = np.array(minVolPortfolio.x)

# Calculate return, volatility & SR. of min. volatility portfolio
minVol_nn_annualised_return = portfolioNonAnnualisedReturn(minVolPortfolio_weights, stockPricesGeneric)
minVol_annualised_return = portfolioAnnualisedReturn(minVolPortfolio_weights, stockPricesGeneric)
minVol_std = portfolioAnnualisedVariance(minVolPortfolio_weights, covMatrixGeneric)
minVolSR = round((minVol_annualised_return - riskFreeRate) / minVol_std, 2)
minVol_nn_annualised_return, minVol_annualised_return, minVol_std = round(minVol_nn_annualised_return*100,2), round(minVol_annualised_return*100,2), round(minVol_std*100,2)

########################################################################
# Step 4: Compute the efficient frontier for the generic stocks
########################################################################
# Define list of target returns for efficient frontier
frontierMinReturn = min(0.5 * (minVol_annualised_return - maxSR_annualised_return), Model_annualised_return, EquWei_annualised_return) / 100
frontierMaxReturn = max(0, 2 * maxSR_annualised_return, maxSR_annualised_return, max(AddStocks_annualised_returns), EquWei_annualised_return) / 100
targetReturns = np.linspace(frontierMinReturn, frontierMaxReturn, num=100)

# Calculate minimum volatility for each target return
volatilityPerTargetReturn = []
for returnTarget in targetReturns:
    volatilityPerTargetReturn.append(efficientOpt(stockPricesGeneric, covMatrixGeneric, returnTarget)['fun'])

# Trim the efficient frontier by removing the vertical parts
volatilityPerTargetReturn, targetReturns = removeDuplicateEnds(volatilityPerTargetReturn, targetReturns)

########################################################################
# Step 5: Plot efficient frontier
# Note: input return & volatility values must be formatted as percentage
########################################################################
efficientFrontierGraph(
    maxSR_annualised_return, maxSR_std, 
    minVol_annualised_return, minVol_std, 
    volatilityPerTargetReturn, targetReturns,
    GenStocks_returns, GenStocks_std,
    AddStocks_annualised_returns, AddStocks_std,
    Model_annualised_return, Model_std, 
    ChatGPT_annualised_return, ChatGPT_std, 
    EquWei_annualised_return, EquWei_std
)

########################################################################
# Step 6: Display returns & variance to web page
########################################################################
# Create a sample DataFrame
dash_data = {
                '-': [
                    Model_Portfolio_Label, 
                    'ChatGPT weighted',
                    'Equally weighted', 
                    'Min. Volatility', 
                    'Max. SR.', 
                    addStockDisplayNames[addStockList[1]], 
                    addStockDisplayNames[addStockList[0]], 
                    addStockDisplayNames[addStockList[2]]
                ],
                '1 Month Return': [
                    str(round(Model_nn_annualised_return,2))+' %', 
                    str(round(ChatGPT_nn_annualised_return,2))+' %', 
                    str(round(EquWei_nn_annualised_return,2))+' %', 
                    str(round(minVol_nn_annualised_return,2))+' %', 
                    str(round(maxSR_nn_annualised_return,2))+' %', 
                    str(round(AddStocks_nn_annualised_returns[1],2))+' %', 
                    str(round(AddStocks_nn_annualised_returns[0],2))+' %', 
                    str(round(AddStocks_nn_annualised_returns[2],2))+' %'
                ],
                'Monthly Annualised Return': [
                    str(Model_annualised_return)+' %', 
                    str(ChatGPT_annualised_return)+' %', 
                    str(EquWei_annualised_return)+' %', 
                    str(minVol_annualised_return)+' %', 
                    str(maxSR_annualised_return)+' %', 
                    str(round(AddStocks_annualised_returns[1],2))+' %', 
                    str(round(AddStocks_annualised_returns[0],2))+' %', 
                    str(round(AddStocks_annualised_returns[2],2))+' %'
                ],
                'Volatility': [
                    str(Model_std)+' %', 
                    str(ChatGPT_std)+' %', 
                    str(EquWei_std)+' %', 
                    str(minVol_std)+' %', 
                    str(maxSR_std)+' %', 
                    str(AddStocks_std[1])+' %', 
                    str(AddStocks_std[0])+' %', 
                    str(AddStocks_std[2])+' %'
                ],
                'Sharpe Ratio':[
                    Model_SR,
                    ChatGPT_SR,
                    EquWeiSR,
                    minVolSR,
                    maxSRportfolioSR,
                    str(AddStocks_SR[1]), 
                    str(AddStocks_SR[0]), 
                    str(AddStocks_SR[2])
                ]
        }
dash_data_df = pd.DataFrame(dash_data)

# Create the Dash web application
app = dash.Dash(__name__)

# Define the layout of the web page
app.layout = html.Div([
    html.H1('Evaluation metrics (daily), '+str(len(Model_Portfolio_Weights))+' stocks, ('+startDate+' to '+endDate+')'),
    DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in dash_data_df.columns],
        data=dash_data_df.to_dict('records')
    ),
    html.Div([
        html.H3('Notes'),
        html.P('Cumulative Return: 100 % + non-annualised return over the analysis period (in %).'),
        html.P('1 Month Return: non-annualised return over the analysis period (1 month) (in %).'),
        html.P('Monthly Annualised Return: (1 + non-annualised return over the analysis period (1 month))**12 - 1 (in %).'),
        html.P('Volatility: annualised daily volatility in trading days.'),
        html.P('Sharpe Ratio: (monthly annualised return % - risk free rate of 0%) / (annualised daily volatility in trading days) (in %).'),
        html.P(f'Weight bounds: {int(100*constraintSet[0])} % ≤ w ≤ {int(100*constraintSet[1])} %.')
    ])
])

# Open the web page in the default browser (Debug true for web page refresh on update)
webbrowser.open('http://127.0.0.1:8050/')
app.run_server(debug=False)