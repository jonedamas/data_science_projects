import yaml
import numpy as np
import yfinance as yf
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt

from utils import *

with open('..\data_science_projects\projects\portfolio_optimization\p_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

portfolio_configs = config_data['PORTFOLIO']

tickers = portfolio_configs['TICKERS']

period = str(portfolio_configs['PERIOD']) + portfolio_configs['PERIOD_UNIT']

stock_data = pd.DataFrame(yf.download(tickers, period=period, ignore_tz=False, progress=False))

closing_price = stock_data['Adj Close']
n = len(tickers)

log_returns = np.log(closing_price/closing_price.shift()).dropna()

mean_returns = np.array(log_returns.mean()) 
cov_matrix = np.array(log_returns.cov())

days = 252

res_matrix_min = np.zeros((days, len(tickers)))
res_matrix_sharpe = np.zeros((days, len(tickers)))

rolling_mean = log_returns.rolling(days).sum().iloc[-days:]

for i in trange(days):

    mean_returns = rolling_mean.iloc[i] * 252

    cov_matrix = log_returns.iloc[-2*days + i:-days+i].cov().to_numpy() * 252 

    OW_min = find_OW(portfolio_var, cov_matrix, n=n)
    res_matrix_min[i, :] = OW_min

    OW_sharpe = find_OW(portfolio_sharpe, cov_matrix, mean_returns, n=n)
    res_matrix_sharpe[i, :] = OW_sharpe

results_min = pd.DataFrame(res_matrix_min, columns=tickers)
results_sharpe = pd.DataFrame(res_matrix_sharpe, columns=tickers)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

results_min.plot(ax=ax1, linewidth=0.9)

ax1.set_title('Minimum variance portfolio')
ax1.set_ylabel('Weight fraction')
ax1.grid(alpha=0.3)

results_sharpe.plot(ax=ax2, linewidth=0.9)

ax2.set_title('Portfolio weights over time sharpe')
ax2.set_ylabel('Weight fraction')
ax2.grid(alpha=0.3)

fig.tight_layout()
plt.show()