import yaml
import numpy as np
import yfinance as yf
import pandas as pd

with open('p_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

portfolio_configs = config_data['PORTFOLIO']

tickers = portfolio_configs['TICKERS']

period = portfolio_configs['PERIOD'] + portfolio_configs['PERIOD_UNIT']

stock_data = pd.DataFrame(yf.download(tickers, period=period, ignore_tz=False, progress=False))

print(stock_data['Adj Close'])