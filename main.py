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
def get_listing_status():
    rate_limiter.wait()
    function = 'LISTING_STATUS'
    url = base_url + '&function={function}'.format(function=function)
    r = requests.get(url)
    decoded_content = r.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    df = pd.DataFrame(my_list[1:],columns=my_list[0:1][0])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df.to_csv('data/{function}.csv'.format(function=function),index=False)
    tickers = pd.read_csv(f'data/{function}.csv'.format(function=function))
    return tickers

def get_overview(ticker):
    rate_limiter.wait()
    function = 'OVERVIEW'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame([data])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df.columns = df.columns.str.replace(r'^(\d)', r'_\1', regex=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_time_series_monthly_adjusted(ticker):
    rate_limiter.wait()
    function = 'TIME_SERIES_MONTHLY_ADJUSTED'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_income_statement(ticker):
    rate_limiter.wait()
    function = 'INCOME_STATEMENT'
    url = f'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['annualReports'])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.to_csv(f'data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_balance_sheet(ticker):
    rate_limiter.wait()
    function = 'BALANCE_SHEET'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['annualReports'])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_cash_flow(ticker):
    rate_limiter.wait()
    function = 'CASH_FLOW'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['annualReports'])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_earnings(ticker):
    rate_limiter.wait()
    function = 'EARNINGS'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['quarterlyEarnings'])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_earnings_estimates(ticker):
    rate_limiter.wait()
    function = 'EARNINGS_ESTIMATES'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    estimates_data = data.get('estimates', [])
    df = pd.DataFrame(estimates_data)# Clean column names (convert camelCase to snake_case)
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.to_csv(f'data/{function}/{ticker}_{function}.csv', index=False)

def get_dividends(ticker):
    rate_limiter.wait()
    function = 'DIVIDENDS'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_splits(ticker):
    rate_limiter.wait()
    function = 'SPLITS'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_shares_outstanding(ticker):
    rate_limiter.wait()
    function = 'SHARES_OUTSTANDING'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_etf_profile(ticker):
    rate_limiter.wait()
    function = 'ETF_PROFILE'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()

    info = {
        'net_assets': [data['net_assets']],
        'net_expense_ratio': [data['net_expense_ratio']],
        'portfolio_turnover': [data['portfolio_turnover']],
        'dividend_yield': [data['dividend_yield']],
        'inception_date': [data['inception_date']],
        'leveraged': [data['leveraged']]
    }

    df = pd.DataFrame(info)
    df['ticker'] = ticker
    df.to_csv('data/{function}/ETF_INFO/{ticker}_ETF_INFO.csv'.format(ticker=ticker,function=function),index=False)

    df = pd.DataFrame(data['sectors'])
    df['ticker'] = ticker
    df.to_csv('data/{function}/ETF_SECTORS/{ticker}_ETF_SECTORS.csv'.format(ticker=ticker,function=function),index=False)

    df = pd.DataFrame(data['holdings'])
    df['ticker'] = ticker
    df.to_csv('data/{function}/ETF_HOLDINGS/{ticker}_ETF_HOLDINGS.csv'.format(ticker=ticker,function=function),index=False)

#%%
# tickers = get_listing_status()
# for testing
tickers = pd.read_json('{"symbol":{"0":"IVV"},"name":{"0":"Agilent Technologies Inc"},"exchange":{"0":"NYSE"},"asset_type":{"0":"ETF"},"ipo_date":{"0":"1999-11-18"},"delisting_date":{"0":null},"status":{"0":"Active"}}')
#%%
stocks = tickers[tickers['type'] == 'Stock']
for ticker in stocks.symbol.unique():
    get_overview(ticker)
    get_time_series_monthly_adjusted(ticker)
    get_income_statement(ticker)
    get_balance_sheet(ticker)
    get_cash_flow(ticker)
    get_earnings(ticker)
    get_earnings_estimates(ticker)
    get_dividends(ticker)
    get_splits(ticker)
    get_shares_outstanding(ticker)
#%%
etfs = tickers[tickers['asset_type'] == 'ETF']
for ticker in etfs.symbol.unique():
    get_time_series_monthly_adjusted(ticker)
    get_dividends(ticker)
    get_etf_profile(ticker)

#%%


