#%%
import pandas as pd
import time
from google.cloud import bigquery
from google.oauth2 import service_account
import config
import requests
import csv
import io
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
#%%
rate_limiter = RateLimiter(calls_per_second=1)

#%% set base url for alpha vantage
apikey = config.alpha_vantage_api_key
base_url = 'https://www.alphavantage.co/query?apikey={apikey}'.format(apikey=apikey)
#%%
# LISTING_STATUS
# function = 'LISTING_STATUS'
# url = base_url + '&function={function}'.format(function=function)
# r = requests.get(url)
# decoded_content = r.content.decode('utf-8')
# cr = csv.reader(decoded_content.splitlines(), delimiter=',')
# my_list = list(cr)
# df = pd.DataFrame(my_list[1:],columns=my_list[0:1][0])
# df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
# df.to_csv('data/{function}.csv'.format(function=function),index=False)
tickers = pd.read_csv(f'data/{function}.csv'.format(function=function))
#%%
ticker = 'IBM'
#%%
#%% TIME_SERIES_MONTHLY_ADJUSTED
function = 'TIME_SERIES_MONTHLY_ADJUSTED'
url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
r = requests.get(url)
df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
df['ticker'] = ticker
df.columns = df.columns.str.replace(' ', '_')
df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

#%%
# INCOME_STATEMENT
function = 'INCOME_STATEMENT'
url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
r = requests.get(url)
data = r.json()
df = pd.DataFrame(data['annualReports'])
df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
df['ticker'] = ticker
df.to_csv(f'data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)
# %%
# BALANCE_SHEET
function = 'BALANCE_SHEET'
url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
r = requests.get(url)
data = r.json()
df = pd.DataFrame(data['annualReports'])
df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
df['ticker'] = ticker
df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)
# %%
# CASH_FLOW
function = 'CASH_FLOW'
url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
r = requests.get(url)
data = r.json()
df = pd.DataFrame(data['annualReports'])
df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
df['ticker'] = ticker
df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

# %%
# EARNINGS
function = 'EARNINGS'
url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
r = requests.get(url)
data = r.json()
df = pd.DataFrame(data['quarterlyEarnings'])
df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
df['ticker'] = ticker
df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)
# %%

