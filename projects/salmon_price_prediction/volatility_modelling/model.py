import pandas as pd
import numpy as np
import arch
from pmdarima import auto_arima

class ARMA_GARCH:
    def __init__(self, data: pd.Series):
        self.data = data

        self.log_yield = np.log(data / data.shift()).dropna() * 100

        # ARMA(1,1) model
        self.arma_model = auto_arima(
            self.log_yield, 
            seasonal=True, 
            stepwise=True, 
            trace=True
        )

        self.arma_order = self.arma_model.get_params()['order']

        # GARCH(1,1) model
        self.garch_model = arch.arch_model(
            self.log_yield, 
            vol='Garch', 
            p=1, q=1, 
            lags=self.arma_order, 
            dist='Normal'
        )

        self.garch_result = self.garch_model.fit()

        self.conditional_volatility = self.garch_result.conditional_volatility

    def summary(self):
        print(self.garch_result.summary())

    def forecast(self, horizon: int):
        return self.garch_result.forecast(horizon=horizon, reindex=False)

