import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_pacf

data = yf.download(tickers='SBIN.NS', period='3000d', interval='1d')
pacf = plot_pacf(data['Close'], lags=25)
plt.plot(data['Close'])
# Stationary Test
# df_stationarityTest = adfuller(data['Close'], autolag='AIC')
# print(df_stationarityTest[1])

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i-interval]
        diff.append(value)

    return np.array(diff)

def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

train = data['Close'][:len(data['Close'])-10]
logged = np.log(train)
differenced = difference(train)
plt.plot(logged)
df_stationarityTest = adfuller(differenced, autolag='AIC')
dl_stationarityTest = adfuller(logged, autolag='AIC')
print(dl_stationarityTest[1])


ar_model = AutoReg(differenced, lags=1).fit()
forecast = ar_model.forecast(steps=10)

history = [x for x in train]
results = list()
day = 1
for yhat in forecast:
    inverted = inverse_difference(history, yhat)
    print('Day %d: %f' % (day, inverted))
    history.append(inverted)
    results.append(inverted)
    day += 1

test = data['Close'].tail(7)
print(test)
res = pd.DataFrame(results)
print(mean_absolute_percentage_error(test,res)*100)


ar_model2 = AutoReg(logged, lags=1).fit()
forecast2 = ar_model2.forecast(steps=10)
history = [x for x in train]
results = list()
day = 1
for yhat in forecast2:
    inverted = np.exp(yhat)
    print('Day %d: %f' % (day, inverted))
    history.append(inverted)
    results.append(inverted)
    day += 1

test = data['Close'].tail(10)
print(test)
print(res)
res = pd.DataFrame(results)
print(mean_absolute_error(test,res))