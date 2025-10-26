#%%
import pandas as pd
import time
from google.cloud import bigquery
from google.oauth2 import service_account
import config
import requests
import csv
import io
import os
import tqdm


credentials = service_account.Credentials.from_service_account_file(config.service_account_path)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

class RateLimiter:
    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
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
rate_limiter = RateLimiter(calls_per_minute=74)

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
    tickers = pd.read_csv('data/{function}.csv'.format(function=function))
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
    df = df.rename(columns={'symbol': 'ticker'})
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_time_series_daily_adjusted(ticker):
    rate_limiter.wait()
    function = 'TIME_SERIES_DAILY_ADJUSTED'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.replace(to_replace='None', value=0, inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_insider_transactions(ticker):
    rate_limiter.wait()
    function = 'INSIDER_TRANSACTIONS'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.DataFrame(r.json()['data'])
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_income_statement(ticker):
    rate_limiter.wait()
    function = 'INCOME_STATEMENT'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['annualReports'])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_balance_sheet(ticker):
    rate_limiter.wait()
    function = 'BALANCE_SHEET'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data['annualReports'])
    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df['ticker'] = ticker
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
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
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
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
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
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
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function), index=False)

def get_dividends(ticker):
    rate_limiter.wait()
    function = 'DIVIDENDS'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_splits(ticker):
    rate_limiter.wait()
    function = 'SPLITS'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_shares_outstanding(ticker):
    rate_limiter.wait()
    function = 'SHARES_OUTSTANDING'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['ticker'] = ticker
    df.columns = df.columns.str.replace(' ', '_')
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=ticker,function=function),index=False)

def get_etf_profile(ticker):
    rate_limiter.wait()
    function = 'ETF_PROFILE'
    url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
    r = requests.get(url)
    data = r.json()

    os.makedirs('data/{function}'.format(function=function), exist_ok=True)

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
    df.replace(to_replace='None', value=0, inplace=True)
    os.makedirs('data/{function}/ETF_INFO'.format(function=function), exist_ok=True)
    df.to_csv('data/{function}/ETF_INFO/{ticker}_ETF_INFO.csv'.format(ticker=ticker,function=function),index=False)

    if data.get('sectors'):
        df = pd.DataFrame(data['sectors'])
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        os.makedirs('data/{function}/ETF_SECTORS'.format(function=function), exist_ok=True)
        df.to_csv('data/{function}/ETF_SECTORS/{ticker}_ETF_SECTORS.csv'.format(ticker=ticker,function=function),index=False)

    if data.get('holdings'):
        df = pd.DataFrame(data['holdings'])
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        os.makedirs('data/{function}/ETF_HOLDINGS'.format(function=function), exist_ok=True)
        df.to_csv('data/{function}/ETF_HOLDINGS/{ticker}_ETF_HOLDINGS.csv'.format(ticker=ticker,function=function),index=False)

#%%
# tickers = get_listing_status()
# stocks = tickers[tickers['asset_type'] == 'Stock']
# etfs = tickers[tickers['asset_type'] == 'ETF']

# for testing
tickers  = pd.read_csv('data/LISTING_STATUS.csv')
stocks = tickers[tickers['asset_type'] == 'Stock'][0:1]
etfs = tickers[tickers['asset_type'] == 'ETF'][0:1]
#%%

for ticker in tqdm.tqdm(stocks.symbol.unique()):
    get_overview(ticker)
    get_time_series_daily_adjusted(ticker)
    get_insider_transactions(ticker)
    get_income_statement(ticker)
    get_balance_sheet(ticker)
    get_cash_flow(ticker)
    get_earnings(ticker)
    get_earnings_estimates(ticker)
    get_dividends(ticker)
    get_splits(ticker)
    get_shares_outstanding(ticker)
    
#%%
for ticker in tqdm.tqdm(etfs.symbol.unique()):
    get_time_series_daily_adjusted(ticker)
    get_dividends(ticker)
    get_etf_profile(ticker)

#%%
commodities = ['WTI','BRENT','NATURAL_GAS','COPPER','ALUMINUM','WHEAT','CORN','COTTON','SUGAR','COFFEE','ALL_COMMODITIES']
for commodity in commodities:
    rate_limiter.wait()
    function = commodity
    url = 'https://www.alphavantage.co/query?function={function}&apikey={apikey}&datatype=csv'.format(function=function,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['commodity'] = commodity
    os.makedirs('data/COMMODITIES'.format(function=function), exist_ok=True)
    df.to_csv('data/COMMODITIES/{function}.csv'.format(function=function),index=False)
    
#%%
economic_indicators = ['REAL_GDP','REAL_GDP_PER_CAPITA','TREASURY_YIELD','FEDERAL_FUNDS_RATE','CPI','INFLATION','RETAIL_SALES','DURABLES','UNEMPLOYMENT','NONFARM_PAYROLL']
for economic_indicator in economic_indicators:
    rate_limiter.wait()
    function = economic_indicator
    url = 'https://www.alphavantage.co/query?function={function}&apikey={apikey}&datatype=csv'.format(function=function,apikey=apikey)
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    df['economic_indicator'] = economic_indicator
    os.makedirs('data/ECONOMIC_INDICATORS'.format(function=function), exist_ok=True)
    df.to_csv('data/ECONOMIC_INDICATORS/{function}.csv'.format(function=function),index=False)

#%%
# import stat
# for root, dirs, files in os.walk("data"):
#     for file in files:
#         file_path = os.path.join(root, file)
#         try:
#             os.chmod(file_path, stat.S_IWRITE)
#             os.remove(file_path)
#             print(f"Deleted: {file}")
#         except Exception as e:
#             print(f"Could not delete {file}: {e}")

#%%
# additional api calls to add
    # add forex rates
    # add technical indicators
    # add earnings call transcripts
    # add news & sentiment
# how to restart where it left off or had an error

get_overview('IVV')
# %%
