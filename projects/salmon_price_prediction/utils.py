import requests
import pandas as pd

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

    return data

def add_lags(df: pd.DataFrame):
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    df['Day of year'] = df.index.dayofyear