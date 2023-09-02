import requests
import pandas as pd

def import_salmon_data(URL: str) -> pd.DataFrame: 
    '''
    Args:
        URL (str): API url for salmon data from ssb

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

    return data

def add_lags(df: pd.DataFrame):
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week.astype(int)
    df['Month'] = df.index.month
    df['Day of year'] = df.index.dayofyear