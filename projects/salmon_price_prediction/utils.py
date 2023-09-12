import requests
import pandas as pd
import xgboost as xgb
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose

def import_salmon_data(URL: str, correct_dt: bool=True, rename_cols: bool=False) -> pd.DataFrame: 
    '''
    Args:
        URL (str): API url for salmon data from ssb

    Kwargs:
        correct_dt (bool): change index to datetime
        rename_cols (bool): rename columns to Price and Volume

    Returns:
        data (DataFrame): Dataframe with price and volume data
    '''
    json_response = requests.get(URL).json()

    data = pd.DataFrame(
        {
            'uke': json_response['dataset']['dimension']['Tid']['category']['label'].values(),
            'Kilopris (kr)': json_response['dataset']['value'][1::2],
            'Vekt (tonn)': json_response['dataset']['value'][::2]
        }
    )

    if correct_dt:
        date_series = pd.to_datetime(data['uke'].str[:4] + data['uke'].str[-2:] + '1', format='%Y%W%w')
        data.set_index(date_series, inplace=True)
        data.drop(['uke'], axis=1, inplace=True)

    if rename_cols:
        data.rename({'Kilopris (kr)':'Price', 'Vekt (tonn)': 'Volume'}, inplace=True, axis='columns')
        data.index.rename('Date', inplace=True)

    return data

def add_lags(df: pd.DataFrame):
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    df['Day of year'] = df.index.dayofyear


def add_decomposition(df: pd.DataFrame, target: str, model: str="multiplicative", period: int=52):
    STL = seasonal_decompose(
        df[target], 
        model=model, 
        period=period
    )

    df['Trend'] = STL.trend
    df['Seasonal'] = STL.seasonal
    df['Residual'] = STL.resid


class xgb_model:
    def __init__(self, data: pd.DataFrame, targets: list[str], features: list[str]):
        self.data = data
        self.targets = targets
        self.features = features

        # Create model
        self.reg_model = xgb.XGBRegressor(
            n_estimators=10000, 
            early_stopping_rounds=1000
        )

        # Create train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data[self.features], 
            self.data[self.targets], 
            test_size=0.30, 
            shuffle=True
        )

        # Fit model
        self.reg_model.fit(
            self.X_train, self.y_train, 
            eval_set=[(self.X_train, self.y_train), (self.X_test, self.y_test)], 
            verbose=False
        )

        self.y_pred = self.reg_model.predict(self.X_test)

    def predict(self, X_pred: pd.DataFrame) -> NDArray[np.float64]:
        return self.reg_model.predict(X_pred)

    def mse(self):
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
