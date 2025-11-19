#%%
class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 76.0 / calls_per_minute
        self.last_call_time = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

#%%     
def call_alpha_vantage(symbol, function_params):
    overwrite=False
    df = pd.DataFrame()

    function = function_params.get('function')
    datatype = function_params.get('datatype')

    url = 'https://www.alphavantage.co/query'
    params = {'apikey': config.alpha_vantage_api_key}
    params.update(function_params)
    params.update({'symbol': symbol})

    filepath = f'data/{function}/{symbol}_{function}.csv'
    if os.path.exists(filepath) and overwrite == False:
        return

    rate_limiter.wait()
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print('Request failed')

    if datatype == 'csv':
        data = r.content.decode('utf-8')
        actual_columns = data.splitlines()[0].split(',')
        if len(actual_columns) <= 1:
            print(f'Data is not in the expected format - {symbol} - {function}'.format(symbol, function))
        df = pd.read_csv(io.StringIO(data))
    else:
        data = r.json()
        if 'annualReports' in data:
            # income statement, balance sheet, cash flow
            df = pd.DataFrame(data['annualReports'])
        elif 'estimates' in data:
            # earnings estimates
            df = pd.DataFrame(data['estimates'])
        elif 'data' in data:
            # insider transactions
            df = pd.DataFrame(data['data'])
        elif 'Symbol' in data and 'AssetType' in data: 
            # overview
            df = pd.DataFrame([data])
        else:
            print(f'Data is not in the expected format - {symbol} - {function}'.format(symbol, function))
            raise Exception(f'Data is not in the expected format - {symbol} - {function}'.format(symbol, function))

    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df.replace(to_replace='None', value=0, inplace=True)
    df['symbol'] = symbol

    # Replace invalid filename characters (< > : " / \ | ? *) with underscores
    symbol = re.sub(r'[<>:"/\\|?*]', '_', symbol)
    df.to_csv(f'data/{function}/{symbol}_{function}.csv', index=False)

#%%
import time
import tqdm
import pandas as pd
import config
import requests
import csv
import os
import io
import re

rate_limiter = RateLimiter(calls_per_minute=76)
rate_limiter.wait()

url = 'https://www.alphavantage.co/query'
params = {'apikey': config.alpha_vantage_api_key}
params.update({'function': 'LISTING_STATUS'})
rate_limiter.wait()
r = requests.get(url, params=params)
data = r.content.decode('utf-8')
df = pd.read_csv(io.StringIO(data))
df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()

#%%
stock_symbols = df[df['asset_type'] == 'Stock']['symbol'].unique()

#%%
for symbol in tqdm.tqdm(stock_symbols):
    try:
        call_alpha_vantage(symbol, {'function': 'OVERVIEW'})
        call_alpha_vantage(symbol, {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'datatype': 'csv'})
    except Exception as e:
        print(f'Error for {symbol}: {e}'.format(symbol, e))

# %%
stock_symbols