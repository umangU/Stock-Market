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

def read_data


model_ar = AutoReg(target_train, lags=2)
model_ar_fit = model_ar.fit()
yhat_ar = model_ar_fit.forecast(steps=218)
yhat_r_df = pd.DataFrame(yhat_ar)
yhat_r_df.index = target_test.index
plot_forecast(target_train, target_test, yhat_r_df)