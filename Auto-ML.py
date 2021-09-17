import numpy as np
import pandas as pd
import yfinance as yf
import pmdarima as pm
import matplotlib.pyplot as plt
import statsmodels.tsa.api as sm
import statsmodels.graphics.tsaplots as smt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_absolute_percentage_error

import warnings

warnings.filterwarnings("ignore")

# Defining Global Functions
def tsplot(y, lags=None, figsize=(12, 7)):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    p_value = sm.adfuller(y)[1]
    ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
    smt.plot_acf(y, lags=lags, ax=acf_ax)
    smt.plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()

def plot_forecast(series_train, series_test, forecast):

    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)

    plt.figure(figsize=(12, 6))
    plt.title(f"MAE: {mae:.2f}, MAPE: {mape:.3f}", size=18)
    plt.plot(series_train, label="train", color="b")
    plt.plot(series_test, label="test", color="g")
    plt.plot(forecast, label="Forecast", color="r")
    plt.legend(loc='best')
    plt.show()

    return mae, mape

def invert_transformation(input, target_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    ip = input
    df_fc = df_forecast.copy()
    columns = input.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (data[col].iloc[-1]-data[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = data[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

# Reading the data
data = yf.download(tickers='BTC-USD', period='3y', interval = '1d')
target = data['Close']
plt.plot(target)
plt.show()

tsplot(target)

# Making the series stationary
target_diff = (target - target.shift(1)).dropna()
tsplot(target_diff)

# Divide the Data into Train and Test
test_len = int(len(target)*0.2)
target_train, target_test = target.iloc[:-test_len], target.iloc[-test_len:]

#################################################################################
# AR Model

model_ar = AutoReg(target_train, lags=2)
model_ar_fit = model_ar.fit()
yhat_ar = model_ar_fit.forecast(steps=218)
yhat_r_df = pd.DataFrame(yhat_ar)
yhat_r_df.index = target_test.index
plot_forecast(target_train, target_test, yhat_r_df)

# MA Model

model_ma = ARMA(target_train, order=(0,1))
model_ma_fit = model_ma.fit(disp=False)
yhat_ma = model_ma_fit.predict(1, len(target_test))
yhat_ma_df = pd.DataFrame(yhat_ma)
yhat_ma_df.index = target_test.index
plot_forecast(target_train, target_test, yhat_ma_df)

# ARMA Model

model_arma = ARMA(target_train, order=(1,0))
model_arma_fit = model_arma.fit(disp=False)
yhat_arma = model_arma_fit.predict(1, len(target_test))
yhat_arma_df = pd.DataFrame(yhat_arma)
yhat_arma_df.index = target_test.index
plot_forecast(target_train, target_test, yhat_arma_df)

# ARIMA model

forecaster = pm.auto_arima(target_train,
                       start_p=0,
                       start_q=0,
                       test='adf',
                       d=None,
                       start_P=0,
                       D=0,
                       seasonal=False,
                       error_action='warn',
                       trace=True,
                       suppress_warnings=True,
                       stepwise=True,
                       random_state=20,
                       n_fits=50)

forecast = forecaster.predict(n_periods = 218, index = target_test.index)
forecast_df = pd.DataFrame(forecast)
forecast_df.index = target_test.index
target_arima_mae, target_arima_mape = plot_forecast(target_train, target_test, forecast_df)

# SARIMA model
forecaster2 = pm.auto_arima(target_train,
                       start_p=1,
                       start_q=1,
                       test='adf',
                       d=None,
                       m=12,
                       start_P=0,
                       D=1,
                       seasonal=True,
                       error_action='warn',
                       trace=True,
                       suppress_warnings=True,
                       stepwise=True,
                       random_state=20,
                       n_fits=50)

sforecast = forecaster2.predict(n_periods = 218, index = target_test.index)
sforecast_df = pd.DataFrame(forecast)
sforecast_df.index = target_test.index
target_sarima_mae, target_sarima_mape = plot_forecast(target_train, target_test, forecast_df)

# VAR Model

nobs = 50
target_train, target_test = data[0:-nobs], data[-nobs:]
train_diff = target_train.diff().dropna()
model = VAR(train_diff)
model_fit =model.fit(1)

lag_order = model_fit.k_ar
forecast_input = train_diff.values[-lag_order:]
fc = model_fit.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=target_test.index, columns = data.columns + '_1d')

df_results = invert_transformation(data, target_train, df_forecast, second_diff=False)
final = df_results.loc[:, ['Open_forecast', 'High_forecast', 'Low_forecast', 'Close_forecast', 'Adj Close_forecast', 'Volume_forecast']]
print(mean_absolute_percentage_error(target_test,final))

plt.figure(figsize=(12, 6))
plt.plot(target_train['Close'], label="train", color="b")
plt.plot(target_test['Close'], label="test", color="g")
plt.plot(final['Close_forecast'], label="Forecast", color="r")
plt.legend(loc='best')
plt.show()

# VMA Model
from statsmodels.tsa.statespace.varmax import VARMAX

vma_subset = data.drop(['Open','Adj Close','Volume'], axis=1)
length = int(len(vma_subset)*0.2)
vma_train, vma_test = vma_subset.iloc[:-length], vma_subset.iloc[-length:]
vmax_train_diff = vma_train.diff().dropna()

    # exog = varmax_train_diff['Open'] -- in case of dependent variable
model_vma = VARMAX(vmax_train_diff[['High','Low','Close']], order=(5,3))
model_vma_fit = model_vma.fit(disp=False)

yhat_vma = model_vma_fit.predict(1,len(vma_test))
yhat_vma_forecast = pd.DataFrame(yhat_vma)
yhat_vma_forecast.index = vma_test.index
yhat_vma_forecast.columns = yhat_vma_forecast.columns.values+'_1d'
yhat_vma_results = invert_transformation(vma_subset, vma_train, yhat_vma_forecast, second_diff=False)
final_vma = yhat_vma_results.loc[:, ['Close_forecast']]
plot_forecast(vma_train['Close'], vma_test['Close'], final_vma)

# VARMA Model
from statsmodels.tsa.statespace.varmax import VARMAX

varma_subset = data.drop(['Open','Adj Close','Volume'], axis=1)
length = int(len(varma_subset)*0.2)
varma_train, varma_test = varma_subset.iloc[:-length], varma_subset.iloc[-length:]
varmax_train_diff = varma_train.diff().dropna()

    # exog = varmax_train_diff['Open'] -- in case of dependent variable
model_varma = VARMAX(varmax_train_diff[['High','Low','Close']], order=(1,1))
model_varma_fit = model_varma.fit(disp=False)

yhat_varma = model_varma_fit.predict(1,len(varma_test))
yhat_varma_forecast = pd.DataFrame(yhat_varma)
yhat_varma_forecast.index = varma_test.index
yhat_varma_forecast.columns = yhat_varma_forecast.columns.values+'_1d'
yhat_varma_results = invert_transformation(varma_subset, varma_train, yhat_varma_forecast, second_diff=False)
final_varma = yhat_varma_results.loc[:, ['Close_forecast']]
plot_forecast(varma_train['Close'], varma_test['Close'], final_varma)


# Single Exponential Smoothing Model
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ses_subset = data['Close']
ses_train, ses_test = ses_subset.iloc[:-test_len], ses_subset.iloc[-test_len:]

ses_model = SimpleExpSmoothing(ses_train)
ses_model_fit = ses_model.fit()
yhat_ses = ses_model_fit.forecast(steps=218)
yhat_ses_df = pd.DataFrame(yhat_ses)
yhat_ses_df.index = ses_test.index
plot_forecast(ses_train, ses_test, yhat_ses_df)

# Double Exponential Smoothing Model
from statsmodels.tsa.holtwinters import ExponentialSmoothing

des_subset = data['Close']
des_train, des_test = des_subset.iloc[:-test_len], des_subset.iloc[-test_len:]

des_model = ExponentialSmoothing(des_train, trend='add', seasonal=None, damped=True)
des_model_fit = des_model.fit()
yhat_des = des_model_fit.forecast(steps=218)
yhat_des_df = pd.DataFrame(yhat_des)
yhat_des_df.index = des_test.index
plot_forecast(des_train, des_test, yhat_des_df)

# Triple Exponential Smoothing Model
from statsmodels.tsa.holtwinters import ExponentialSmoothing

tes_subset = data['Close']
tes_train, tes_test = tes_subset.iloc[:-test_len], tes_subset.iloc[-test_len:]

tes_model = ExponentialSmoothing(tes_train, trend="add", seasonal="add", seasonal_periods=7, damped=True)
tes_model_fit = tes_model.fit()
yhat_tes = tes_model_fit.forecast(steps=218)
yhat_tes_df = pd.DataFrame(yhat_tes)
yhat_tes_df.index = tes_test.index
plot_forecast(tes_train, tes_test, yhat_tes_df)