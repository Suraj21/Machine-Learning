# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 23:40:45 2019

@author: suraj
"""

# Import the library we need, which is Pandas and Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Import statsmodel
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller


# Set some parameters to get good visuals - style to ggplot and size to 15,10
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (15, 10)


# Read the csv file of Monthwise Quantity and Price csv file we have.
df = pd.read_csv('D:/GitRepository/Machine-Learning/TimeSeriesAnalysis/MonthWiseMarketArrivals_clean.csv')


# Changing the date column to a Time Interval columnn
df.date = pd.DatetimeIndex(df.date)


# Change the index to the date column
df.index = pd.PeriodIndex(df.date, freq='M')


#ihort the data frame by date
df = df.sort_values(by = "date")


df.head()


dfBang = df.loc[df.city == "BANGALORE"].copy()

dfBang.head()


# Drop redundant columns
dfBang = dfBang.drop(["market", "month", "year", "state", "city", "priceMin", "priceMax"], axis = 1)

dfBang.head()

dfBang.priceMod.plot()


dfBang.quantity.plot()


dfBang.priceMod.plot(kind = "hist", bins = 30)

dfBang['priceModLog'] = np.log(dfBang.priceMod)
dfBang.head()

dfBang.priceModLog.plot(kind = "hist", bins = 30)

dfBang.priceModLog.plot()

"""
Basic Time Series Model¶
We will build a time-series forecasting model to get a forecast for Onion prices. Let us start with the three most basic models -

Mean Constant Model
Linear Trend Model
Random Walk Model
"""



model_mean_pred = dfBang.priceModLog.mean()


# Let us store this as our Mean Predication Value
dfBang["priceMean"] = np.exp(model_mean_pred)

dfBang.plot(kind="line", x="date", y = ["priceMod", "priceMean"])


"""
Can we measure the error rate?

We will use Root Mean Squared Error (RMSE) to calculate our error values

$RMSE = \Sigma \sqrt{ (\hat{y} - y)^2/n} $ , where $\hat{y}$ is predicted value of y
"""
def RMSE(predicted, actual):
    mse = (predicted - actual)**2
    rmse = np.sqrt(mse.sum()/mse.count())
    return rmse

model_mean_RMSE = RMSE(dfBang.priceMean, dfBang.priceMod)
model_mean_RMSE

# Save this in a dataframe
dfBangResults = pd.DataFrame(columns = ["Model", "Forecast", "RMSE"])
dfBangResults.head()


dfBangResults.loc[0,"Model"] = "Mean"
dfBangResults.loc[0,"Forecast"] = np.exp(model_mean_pred)
dfBangResults.loc[0,"RMSE"] = model_mean_RMSE
dfBangResults.head()

"""
Linear Trend Model
"""
dfBang.head()
dfBang.dtypes
# What is the starting month of our data
dfBang.date.min()


# Convert date in datetimedelta figure starting from zero
dfBang["timeIndex"] = dfBang.date - dfBang.date.min()

dfBang.head()

dfBang.dtypes


# Convert to months using the timedelta function
dfBang["timeIndex"] =  dfBang["timeIndex"]/np.timedelta64(1, 'M')

dfBang.timeIndex.head()


# Round the number to 0
dfBang["timeIndex"] = dfBang["timeIndex"].round(0).astype(int)

dfBang.timeIndex.tail()

dfBang.head()

## Now plot linear regression between priceMod and timeIndex
model_linear = smf.ols('priceModLog ~ timeIndex', data = dfBang).fit()

model_linear.summary()


## Parameters for y = mx + c equation
model_linear.params

c = model_linear.params[0]
c

m = model_linear.params[1]
m

model_linear_pred = model_linear.predict()


model_linear_pred


# Plot the prediction line
dfBang.plot(kind="line", x="timeIndex", y = "priceModLog")
plt.plot(dfBang.timeIndex,model_linear_pred, '-')

model_linear.resid.plot(kind = "bar")

# Manual Calculation
model_linear_forecast_manual = m * 146 + c
model_linear_forecast_manual

# Using Predict Function
model_linear_forecast_auto = model_linear.predict(exog = dict(timeIndex=int(146)))
model_linear_forecast_auto

dfBang["priceLinear"] = np.exp(model_linear_pred)

# Root Mean Squared Error (RMSE)
model_linear_RMSE = RMSE(dfBang.priceLinear, dfBang.priceMod)
model_linear_RMSE


dfBangResults.loc[1,"Model"] = "Linear"
dfBangResults.loc[1,"Forecast"] = np.exp(model_linear_forecast_manual)
dfBangResults.loc[1,"RMSE"] = model_linear_RMSE
dfBangResults.head()


dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear"])

"""
Linear Model with Regressor
"""


## Now plot linear regression between priceMod and timeIndex
model_linear_quantity = smf.ols('priceModLog ~ timeIndex + np.log(quantity)', data = dfBang).fit()

model_linear_quantity.summary()

dfBang["priceLinearQuantity"] = np.exp(model_linear_quantity.predict())


dfBang.plot(kind = "line", x="timeIndex", y = "quantity")
dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", 
                                             "priceLinear", "priceLinearQuantity"])
    
"""
Random Walk Model
"""


dfBang["priceModLogShift1"] = dfBang.priceModLog.shift()

dfBang.head()


dfBang.plot(kind= "scatter", y = "priceModLog", x = "priceModLogShift1", s = 50)

# Lets plot the one-month difference curve
dfBang["priceModLogDiff"] = dfBang.priceModLog - dfBang.priceModLogShift1


dfBang.priceModLogDiff.plot()

dfBang["priceRandom"] = np.exp(dfBang.priceModLogShift1)
dfBang.head()

dfBang.tail()


dfBang.plot(kind="line", x="timeIndex", y = ["priceMod","priceRandom"])

# Root Mean Squared Error (RMSE)
model_random_RMSE = RMSE(dfBang.priceRandom, dfBang.priceMod)
model_random_RMSE

dfBangResults.loc[2,"Model"] = "Random"
dfBangResults.loc[2,"Forecast"] = np.exp(dfBang.priceModLogShift1[-1])
dfBangResults.loc[2,"RMSE"] = model_random_RMSE
dfBangResults.head()

dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear", "priceRandom"])

"""
Augmented Dickey Fuller Test of Stationarity
"""
def adf(ts):
    
    # Determing rolling statistics
    rolmean = pd.rolling_mean(ts, window=12)
    rolstd = pd.rolling_std(ts, window=12)

    #Plot rolling statistics:
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Calculate ADF factors
    adftest = adfuller(ts, autolag='AIC')
    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',
                                              'Number of Observations Used'])
    for key,value in adftest[4].items():
        adfoutput['Critical Value (%s)'%key] = value
    return adfoutput


"""
Simple Moving Average
"""

# For smoothing the values we can use 12 month Moving Averages 
dfBang['priceModLogMA12'] = pd.rolling_mean(dfBang.priceModLog, window = 12)

dfBang.plot(kind ="line", y=["priceModLogMA12", "priceModLog"])

dfBang["priceMA12"] = np.exp(dfBang.priceModLogMA12)
dfBang.tail()


model_MA12_forecast = dfBang.priceModLog.tail(12).mean()

# Root Mean Squared Error (RMSE)
model_MA12_RMSE = RMSE(dfBang.priceMA12, dfBang.priceMod)
model_MA12_RMSE


dfBangResults.loc[3,"Model"] = "Moving Average 12"
dfBangResults.loc[3,"Forecast"] = np.exp(model_MA12_forecast)
dfBangResults.loc[3,"RMSE"] = model_MA12_RMSE
dfBangResults.head()

dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear",
                                             "priceRandom", "priceMA12"])
    
# Test remaining part for Stationary
ts = dfBang.priceModLog - dfBang.priceModLogMA12
ts.dropna(inplace = True)
adf(ts)

"""

Simple Exponential Smoothing Model (SES)
formula=  alpha * y_exp + (1 - alpha) * y_for
"""

dfBang['priceModLogExp12'] = pd.ewma(dfBang.priceModLog, halflife=12)

halflife = 12
alpha = 1 - np.exp(np.log(0.5)/halflife)
alpha

dfBang.plot(kind ="line", y=["priceModLogExp12", "priceModLog"])

dfBang["priceExp12"] = np.exp(dfBang.priceModLogExp12)
dfBang.tail()

# Root Mean Squared Error (RMSE)
model_Exp12_RMSE = RMSE(dfBang.priceExp12, dfBang.priceMod)
model_Exp12_RMSE


y_exp = dfBang.priceModLog[-1]
y_exp

y_for = dfBang.priceModLogExp12[-1]
y_for


model_Exp12_forecast = alpha * y_exp + (1 - alpha) * y_for

dfBangResults.loc[4,"Model"] = "Exp Smoothing 12"
dfBangResults.loc[4,"Forecast"] = np.exp(model_Exp12_forecast)
dfBangResults.loc[4,"RMSE"] = model_Exp12_RMSE
dfBangResults.head()

dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear", 
                                             "priceRandom",
                                             "priceMA12", "priceExp12"])
    

# Test remaining part for Stationary
ts = dfBang.priceModLog - dfBang.priceModLogExp12
ts.dropna(inplace = True)
adf(ts)

"""
Eliminating Trend and Seasonality
Differencing
"""

dfBang.priceModLogDiff.plot()

# Test remaining part for Stationary
ts = dfBang.priceModLogDiff
ts.dropna(inplace = True)
adf(ts)

from statsmodels.tsa.seasonal import seasonal_decompose
dfBang.index = dfBang.index.to_datetime()
dfBang.head()


decomposition = seasonal_decompose(dfBang.priceModLog, model = "additive")
decomposition.plot()

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid


dfBang["priceDecomp"] = np.exp(trend + seasonal)

# Root Mean Squared Error (RMSE)
model_Decomp_RMSE = RMSE(dfBang.priceDecomp, dfBang.priceMod)
model_Decomp_RMSE


dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear", "priceRandom",
                                             "priceMA12", "priceExp12", "priceDecomp"])
    
dfBang.plot(kind="line", x="timeIndex", y = ["priceMod",
                                              "priceDecomp"])
    

# Test remaining part for Stationary
ts = decomposition.resid
ts.dropna(inplace = True)
adf(ts)


"""
Auto Regressive Models - AR(p)
In an autoregression model, we forecast the variable of interest using a linear combination of past values of the variable. The term autoregression indicates that it is a regression of the variable against itself.

Thus an autoregressive model of order (p) can be written as

$$ y_t = c + m_1y_{t-1} + m_2y_{t-2} + m_3y_{t-3} + .. \\$$
Random walk model is an AR(1) model with $$m_1=1, c = 0\\$$ Random walk model with drift model $$m_1=1, c \not= 0\\$$

We normally restrict autoregressive models to stationary data, and then some constraints on the values of the parameters are required.

For an AR(1) model: $$ −1

Moving Average Model - MA(q)
Rather than use past values of the forecast variable in a regression, a moving average model uses past forecast errors in a regression-like model.

$$ y_t=c+e_t+l_1 e_{t−1}+l_2 e_{t−2} + ... + l_q e_{t-q} \\$$
where e is white noise. We refer to this as an MA(q) model. Of course, we do not observe the values of e(t), so it is not really regression in the usual sense.

Notice that each value of y(t) can be thought of as a weighted moving average of the past few forecast errors. However, moving average models should not be confused with moving average smoothing. A moving average model is used for forecasting future values while moving average smoothing is used for estimating the trend-cycle of past values.

ARIMA Model
If we combine differencing with autoregression and a moving average model, we obtain a non-seasonal ARIMA model. ARIMA is an acronym for AutoRegressive Integrated Moving Average model (“integration” in this context is the reverse of differencing). The full model can be written as

Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. For instance if p is 5, the predictors for y(t) will be y(t-1)….y(t-5).
Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for y(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.
Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.
An importance concern here is how to determine the value of ‘p’ and ‘q’. We use two plots to determine these numbers. Lets discuss them first.

Autocorrelation Function (ACF): It is a measure of the correlation between the the TS with a lagged version of itself. For instance at lag 5, ACF would compare series at time instant ‘t1’…’t2’ with series at instant ‘t1-5’…’t2-5’ (t1-5 and t2 being end points).
Partial Autocorrelation Function (PACF): This measures the correlation between the TS with a lagged version of itself but after eliminating the variations already explained by the intervening comparisons. Eg at lag 5, it will check the correlation but remove the effects already explained by lags 1 to 4.
In MA model, noise / shock quickly vanishes with time. The AR model has a much lasting effect of the shock.
"""

ts = dfBang.priceModLog
ts_diff = dfBang.priceModLogDiff
ts_diff.dropna(inplace = True)



#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_diff, nlags=20)

lag_acf

ACF = pd.Series(lag_acf)

ACF


ACF.plot(kind = "bar")

lag_pacf = pacf(ts_diff, nlags=20, method='ols')

"""
Running the ARIMA Model
"""

from statsmodels.tsa.arima_model import ARIMA 


ts_diff.head()


# Running the ARIMA Model(1,0,1)
model_AR1MA = ARIMA(ts_diff, order=(1,0,1))

results_ARIMA = model_AR1MA.fit(disp = -1)

results_ARIMA.fittedvalues.head()


ts_diff.plot()
results_ARIMA.fittedvalues.plot()

ts_diff.sum()

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff.tail()

predictions_ARIMA_diff.sum()

predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.tail()

ts.ix[0]


predictions_ARIMA_log = pd.Series(ts.ix[0], index=ts.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.tail()


dfBang['priceARIMA'] = np.exp(predictions_ARIMA_log)

dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceARIMA"])


dfBang.plot(kind="line", x="timeIndex", y = ["priceMod", "priceMean", "priceLinear", "priceRandom",
                                             "priceMA12", "priceExp12", "priceDecomp", "priceARIMA"])
    










