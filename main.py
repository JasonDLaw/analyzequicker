#%%
import pandas as pd
import time
from google.cloud import bigquery
from google.oauth2 import service_account
import config
import requests
import csv
credentials = service_account.Credentials.from_service_account_file(config.service_account_path)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

class RateLimiter:
    def __init__(self, calls_per_second):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()

rate_limiter = RateLimiter(calls_per_second=1)

#%% set base url for alpha vantage
apikey = config.alpha_vantage_api_key
base_url = 'https://www.alphavantage.co/query?apikey={apikey}'.format(apikey=apikey)
#%%
# import tickers from alpha vantage
# function = 'LISTING_STATUS'
# url = base_url + '&function={function}'.format(function=function)

# with requests.Session() as s:
#     download = s.get(url)
#     decoded_content = download.content.decode('utf-8')
#     cr = csv.reader(decoded_content.splitlines(), delimiter=',')
#     my_list = list(cr)
#     df = pd.DataFrame(my_list[1:],columns=my_list[0:1][0])
#     df.to_csv('data/tickers.csv'.format(function=function),index=False)
# tickers = pd.read_csv('data/tickers.csv')

#%%
ticker = 'IBM'
#%%
function = 'TIME_SERIES_MONTHLY_ADJUSTED'
url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
r = requests.get(url)
data = r.json()

time_series_data = data['Monthly Adjusted Time Series']
symbol = data['Meta Data']['2. Symbol']

df = pd.DataFrame.from_dict(time_series_data, orient='index')
df['symbol'] = symbol
df['date'] = df.index
df = df.reset_index(drop=True)

df.columns = df.columns.str.replace(r'[0-9.]+\s*', '', regex=True)  # Remove numbers and periods
df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
df.columns = df.columns.str.lower()  # Convert to lowercase for consistency

df.to_csv('data/TIME_SERIES_MONTHLY_ADJUSTED/{ticker}_TIME_SERIES_MONTHLY_ADJUSTED.csv'.format(ticker=ticker),index=False)

#%%



