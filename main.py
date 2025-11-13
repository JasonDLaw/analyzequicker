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
import re
import logging
from datetime import datetime

# Set up logging to file
os.makedirs('logs', exist_ok=True)
log_filename = f'logs/errors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

#%%
credentials = service_account.Credentials.from_service_account_file(config.service_account_path)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

def sanitize_filename(filename):
    """Replace invalid filename characters with underscores"""
    # Windows invalid characters: < > : " / \ | ? *
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

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
    try:
        rate_limiter.wait()
        function = 'OVERVIEW'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame([data])
        df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
        df.columns = df.columns.str.replace(r'^(\d)', r'_\1', regex=True)
        df = df.rename(columns={'symbol': 'ticker'})
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting overview for {ticker}: {e}", exc_info=True)

def get_time_series_daily_adjusted(ticker):
    try:
        function = 'TIME_SERIES_DAILY_ADJUSTED'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        df.columns = df.columns.str.replace(' ', '_')
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting time series daily adjusted for {ticker}: {e}", exc_info=True)


def get_insider_transactions(ticker):
    try:
        function = 'INSIDER_TRANSACTIONS'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        df = pd.DataFrame(r.json()['data'])
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting insider transactions for {ticker}: {e}", exc_info=True)

def get_income_statement(ticker):
    try:
        function = 'INCOME_STATEMENT'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['annualReports'])
        df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting income statement for {ticker}: {e}", exc_info=True)

def get_balance_sheet(ticker):
    try:
        function = 'BALANCE_SHEET'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['annualReports'])
        df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting balance sheet for {ticker}: {e}", exc_info=True)

def get_cash_flow(ticker):
    try:
        function = 'CASH_FLOW'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['annualReports'])
        df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting cash flow for {ticker}: {e}", exc_info=True)

def get_earnings(ticker):
    try:
        function = 'EARNINGS'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        data = r.json()
        df = pd.DataFrame(data['quarterlyEarnings'])
        df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting earnings for {ticker}: {e}", exc_info=True)

def get_earnings_estimates(ticker):
    try:
        function = 'EARNINGS_ESTIMATES'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        data = r.json()
        estimates_data = data.get('estimates', [])
        df = pd.DataFrame(estimates_data)# Clean column names (convert camelCase to snake_case)
        df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
        df['ticker'] = ticker
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function), index=False)
    except Exception as e:
        logging.error(f"Error getting earnings estimates for {ticker}: {e}", exc_info=True)

def get_dividends(ticker):
    try:
        function = 'DIVIDENDS'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df['ticker'] = ticker
        df.columns = df.columns.str.replace(' ', '_')
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting dividends for {ticker}: {e}", exc_info=True)

def get_splits(ticker):
    try:
        function = 'SPLITS'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df['ticker'] = ticker
        df.columns = df.columns.str.replace(' ', '_')
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting splits for {ticker}: {e}", exc_info=True)

def get_shares_outstanding(ticker):
    try:
        function = 'SHARES_OUTSTANDING'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        if os.path.exists('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
        url = 'https://www.alphavantage.co/query?function={function}&symbol={ticker}&apikey={apikey}&datatype=csv'.format(function=function,ticker=ticker,apikey=apikey)
        r = requests.get(url)
        df = pd.read_csv(io.StringIO(r.content.decode('utf-8')))
        df['ticker'] = ticker
        df.columns = df.columns.str.replace(' ', '_')
        df.replace(to_replace='None', value=0, inplace=True)
        df.to_csv('data/{function}/{ticker}_{function}.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting shares outstanding for {ticker}: {e}", exc_info=True)

def get_etf_profile(ticker):
    try:
        function = 'ETF_PROFILE'
        os.makedirs('data/{function}'.format(function=function), exist_ok=True)
        safe_ticker = sanitize_filename(ticker)
        
        # Check if main ETF_INFO file exists (sectors and holdings are optional)
        if os.path.exists('data/{function}/ETF_INFO/{ticker}_ETF_INFO.csv'.format(ticker=safe_ticker,function=function)):
            return

        rate_limiter.wait()
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
        df.replace(to_replace='None', value=0, inplace=True)
        os.makedirs('data/{function}/ETF_INFO'.format(function=function), exist_ok=True)
        df.to_csv('data/{function}/ETF_INFO/{ticker}_ETF_INFO.csv'.format(ticker=safe_ticker,function=function),index=False)

        if data.get('sectors'):
            df = pd.DataFrame(data['sectors'])
            df['ticker'] = ticker
            df.replace(to_replace='None', value=0, inplace=True)
            os.makedirs('data/{function}/ETF_SECTORS'.format(function=function), exist_ok=True)
            df.to_csv('data/{function}/ETF_SECTORS/{ticker}_ETF_SECTORS.csv'.format(ticker=safe_ticker,function=function),index=False)

        if data.get('holdings'):
            df = pd.DataFrame(data['holdings'])
            df['ticker'] = ticker
            df.replace(to_replace='None', value=0, inplace=True)
            os.makedirs('data/{function}/ETF_HOLDINGS'.format(function=function), exist_ok=True)
            df.to_csv('data/{function}/ETF_HOLDINGS/{ticker}_ETF_HOLDINGS.csv'.format(ticker=safe_ticker,function=function),index=False)
    except Exception as e:
        logging.error(f"Error getting ETF profile for {ticker}: {e}", exc_info=True)

#%%
tickers = get_listing_status()
stocks = tickers[tickers['asset_type'] == 'Stock']
etfs = tickers[tickers['asset_type'] == 'ETF']

# for testing
# stocks = pd.DataFrame(['IBM'], columns=['symbol'])
# etfs = pd.DataFrame(['SPY'], columns=['symbol'])

#%%
for ticker in tqdm.tqdm(stocks.symbol.unique()):
    get_time_series_daily_adjusted(ticker)
    get_overview(ticker)
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
# load data to bigquery
# additional api calls to add
    # add forex rates
    # add technical indicators
    # add earnings call transcripts
    # add news & sentiment
# how to restart where it left off or had an error

#%%
listing_status_df = pd.read_csv('data/LISTING_STATUS.csv')
listing_status_df['ipo_date'] = pd.to_datetime(listing_status_df['ipo_date'].replace('null', None), errors='coerce')
listing_status_df['delisting_date'] = pd.to_datetime(listing_status_df['delisting_date'].replace('null', None), errors='coerce')

schema = [
    bigquery.SchemaField("symbol", "STRING"),
    bigquery.SchemaField("name", "STRING"), 
    bigquery.SchemaField("exchange", "STRING"),
    bigquery.SchemaField("asset_type", "STRING"),
    bigquery.SchemaField("ipo_date", "DATE"),
    bigquery.SchemaField("delisting_date", "DATE"),
    bigquery.SchemaField("status", "STRING")
]

job = client.load_table_from_dataframe(
    listing_status_df,
    "analyzequicker.analyzequicker.listing_status",
    job_config=bigquery.LoadJobConfig(schema=schema,write_disposition="WRITE_TRUNCATE")
)
job.result()  # Wait for the job to complete

#%%
# Combine and load BALANCE_SHEET data to BigQuery
import glob

#%%
time_series_files = glob.glob('data/TIME_SERIES_DAILY_ADJUSTED/*_TIME_SERIES_DAILY_ADJUSTED.csv')

# Define BigQuery schema for time series
time_series_schema = [
    bigquery.SchemaField("timestamp", "DATE"),
    bigquery.SchemaField("open", "FLOAT64"),
    bigquery.SchemaField("high", "FLOAT64"),
    bigquery.SchemaField("low", "FLOAT64"),
    bigquery.SchemaField("close", "FLOAT64"),
    bigquery.SchemaField("adjusted_close", "FLOAT64"),
    bigquery.SchemaField("volume", "FLOAT64"),
    bigquery.SchemaField("dividend_amount", "FLOAT64"),
    bigquery.SchemaField("split_coefficient", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

BATCH_SIZE = 1000  # Process 100 files at a time
total_rows = 0

print(f"Processing {len(time_series_files)} files in batches of {BATCH_SIZE}...")

for i in range(0, len(time_series_files), BATCH_SIZE):
    batch_files = time_series_files[i:i + BATCH_SIZE]
    time_series_dfs = []
    
    for file in batch_files:
        df = pd.read_csv(file)
        time_series_dfs.append(df)
    
    time_series_df = pd.concat(time_series_dfs, ignore_index=True)
    
    # Convert date columns
    time_series_df['timestamp'] = pd.to_datetime(time_series_df['timestamp'], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
    for col in numeric_columns:
        time_series_df[col] = pd.to_numeric(time_series_df[col], errors='coerce')
    
    # Use WRITE_TRUNCATE for first batch, WRITE_APPEND for subsequent batches
    write_disposition = "WRITE_TRUNCATE" if i == 0 else "WRITE_APPEND"
    
    job = client.load_table_from_dataframe(
        time_series_df,
        "analyzequicker.analyzequicker.time_series_daily_adjusted",
        job_config=bigquery.LoadJobConfig(schema=time_series_schema, write_disposition=write_disposition)
    )
    job.result()  # Wait for the job to complete
    
    total_rows += len(time_series_df)
    print(f"Batch {i//BATCH_SIZE + 1}: Loaded {len(time_series_df)} rows (Total: {total_rows})")
    
    # Clear memory
    del time_series_dfs, time_series_df

print(f"✅ Loaded {total_rows} total rows to time_series_daily_adjusted table")

#%%
balance_sheet_files = glob.glob('data/BALANCE_SHEET/*_BALANCE_SHEET.csv')
balance_sheet_dfs = []

for file in balance_sheet_files:
    df = pd.read_csv(file)
    balance_sheet_dfs.append(df)

balance_sheet_df = pd.concat(balance_sheet_dfs, ignore_index=True)

# Convert date columns
balance_sheet_df['fiscal_date_ending'] = pd.to_datetime(balance_sheet_df['fiscal_date_ending'], errors='coerce')

# Convert numeric columns (all except ticker, reported_currency, and fiscal_date_ending)
numeric_columns = balance_sheet_df.columns.difference(['fiscal_date_ending', 'reported_currency', 'ticker'])
for col in numeric_columns:
    balance_sheet_df[col] = pd.to_numeric(balance_sheet_df[col], errors='coerce')

# Define BigQuery schema for balance sheet
balance_sheet_schema = [
    bigquery.SchemaField("fiscal_date_ending", "DATE"),
    bigquery.SchemaField("reported_currency", "STRING"),
    bigquery.SchemaField("total_assets", "FLOAT64"),
    bigquery.SchemaField("total_current_assets", "FLOAT64"),
    bigquery.SchemaField("cash_and_cash_equivalents_at_carrying_value", "FLOAT64"),
    bigquery.SchemaField("cash_and_short_term_investments", "FLOAT64"),
    bigquery.SchemaField("inventory", "FLOAT64"),
    bigquery.SchemaField("current_net_receivables", "FLOAT64"),
    bigquery.SchemaField("total_non_current_assets", "FLOAT64"),
    bigquery.SchemaField("property_plant_equipment", "FLOAT64"),
    bigquery.SchemaField("accumulated_depreciation_amortization_ppe", "FLOAT64"),
    bigquery.SchemaField("intangible_assets", "FLOAT64"),
    bigquery.SchemaField("intangible_assets_excluding_goodwill", "FLOAT64"),
    bigquery.SchemaField("goodwill", "FLOAT64"),
    bigquery.SchemaField("investments", "FLOAT64"),
    bigquery.SchemaField("long_term_investments", "FLOAT64"),
    bigquery.SchemaField("short_term_investments", "FLOAT64"),
    bigquery.SchemaField("other_current_assets", "FLOAT64"),
    bigquery.SchemaField("other_non_current_assets", "FLOAT64"),
    bigquery.SchemaField("total_liabilities", "FLOAT64"),
    bigquery.SchemaField("total_current_liabilities", "FLOAT64"),
    bigquery.SchemaField("current_accounts_payable", "FLOAT64"),
    bigquery.SchemaField("deferred_revenue", "FLOAT64"),
    bigquery.SchemaField("current_debt", "FLOAT64"),
    bigquery.SchemaField("short_term_debt", "FLOAT64"),
    bigquery.SchemaField("total_non_current_liabilities", "FLOAT64"),
    bigquery.SchemaField("capital_lease_obligations", "FLOAT64"),
    bigquery.SchemaField("long_term_debt", "FLOAT64"),
    bigquery.SchemaField("current_long_term_debt", "FLOAT64"),
    bigquery.SchemaField("long_term_debt_noncurrent", "FLOAT64"),
    bigquery.SchemaField("short_long_term_debt_total", "FLOAT64"),
    bigquery.SchemaField("other_current_liabilities", "FLOAT64"),
    bigquery.SchemaField("other_non_current_liabilities", "FLOAT64"),
    bigquery.SchemaField("total_shareholder_equity", "FLOAT64"),
    bigquery.SchemaField("treasury_stock", "FLOAT64"),
    bigquery.SchemaField("retained_earnings", "FLOAT64"),
    bigquery.SchemaField("common_stock", "FLOAT64"),
    bigquery.SchemaField("common_stock_shares_outstanding", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    balance_sheet_df,
    "analyzequicker.analyzequicker.balance_sheet",
    job_config=bigquery.LoadJobConfig(schema=balance_sheet_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()  # Wait for the job to complete
print(f"Loaded {len(balance_sheet_df)} rows to balance_sheet table")

#%%
# Combine and load CASH_FLOW data to BigQuery
cash_flow_files = glob.glob('data/CASH_FLOW/*_CASH_FLOW.csv')
cash_flow_dfs = []

for file in cash_flow_files:
    df = pd.read_csv(file)
    cash_flow_dfs.append(df)

cash_flow_df = pd.concat(cash_flow_dfs, ignore_index=True)

# Convert date columns
cash_flow_df['fiscal_date_ending'] = pd.to_datetime(cash_flow_df['fiscal_date_ending'], errors='coerce')

# Convert numeric columns (all except ticker, reported_currency, and fiscal_date_ending)
numeric_columns = cash_flow_df.columns.difference(['fiscal_date_ending', 'reported_currency', 'ticker'])
for col in numeric_columns:
    cash_flow_df[col] = pd.to_numeric(cash_flow_df[col], errors='coerce')

# Define BigQuery schema for cash flow
cash_flow_schema = [
    bigquery.SchemaField("fiscal_date_ending", "DATE"),
    bigquery.SchemaField("reported_currency", "STRING"),
    bigquery.SchemaField("operating_cashflow", "FLOAT64"),
    bigquery.SchemaField("payments_for_operating_activities", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_operating_activities", "FLOAT64"),
    bigquery.SchemaField("change_in_operating_liabilities", "FLOAT64"),
    bigquery.SchemaField("change_in_operating_assets", "FLOAT64"),
    bigquery.SchemaField("depreciation_depletion_and_amortization", "FLOAT64"),
    bigquery.SchemaField("capital_expenditures", "FLOAT64"),
    bigquery.SchemaField("change_in_receivables", "FLOAT64"),
    bigquery.SchemaField("change_in_inventory", "FLOAT64"),
    bigquery.SchemaField("profit_loss", "FLOAT64"),
    bigquery.SchemaField("cashflow_from_investment", "FLOAT64"),
    bigquery.SchemaField("cashflow_from_financing", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_repayments_of_short_term_debt", "FLOAT64"),
    bigquery.SchemaField("payments_for_repurchase_of_common_stock", "FLOAT64"),
    bigquery.SchemaField("payments_for_repurchase_of_equity", "FLOAT64"),
    bigquery.SchemaField("payments_for_repurchase_of_preferred_stock", "FLOAT64"),
    bigquery.SchemaField("dividend_payout", "FLOAT64"),
    bigquery.SchemaField("dividend_payout_common_stock", "FLOAT64"),
    bigquery.SchemaField("dividend_payout_preferred_stock", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_issuance_of_common_stock", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_issuance_of_long_term_debt_and_capital_securities_net", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_issuance_of_preferred_stock", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_repurchase_of_equity", "FLOAT64"),
    bigquery.SchemaField("proceeds_from_sale_of_treasury_stock", "FLOAT64"),
    bigquery.SchemaField("change_in_cash_and_cash_equivalents", "FLOAT64"),
    bigquery.SchemaField("change_in_exchange_rate", "FLOAT64"),
    bigquery.SchemaField("net_income", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    cash_flow_df,
    "analyzequicker.analyzequicker.cash_flow",
    job_config=bigquery.LoadJobConfig(schema=cash_flow_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()  # Wait for the job to complete
print(f"Loaded {len(cash_flow_df)} rows to cash_flow table")

#%%
# Combine and load INCOME_STATEMENT data to BigQuery
income_statement_files = glob.glob('data/INCOME_STATEMENT/*_INCOME_STATEMENT.csv')
income_statement_dfs = []

for file in income_statement_files:
    df = pd.read_csv(file)
    income_statement_dfs.append(df)

income_statement_df = pd.concat(income_statement_dfs, ignore_index=True)

# Convert date columns
income_statement_df['fiscal_date_ending'] = pd.to_datetime(income_statement_df['fiscal_date_ending'], errors='coerce')

# Convert numeric columns
numeric_columns = income_statement_df.columns.difference(['fiscal_date_ending', 'reported_currency', 'ticker'])
for col in numeric_columns:
    income_statement_df[col] = pd.to_numeric(income_statement_df[col], errors='coerce')

# Define BigQuery schema for income statement
income_statement_schema = [
    bigquery.SchemaField("fiscal_date_ending", "DATE"),
    bigquery.SchemaField("reported_currency", "STRING"),
    bigquery.SchemaField("gross_profit", "FLOAT64"),
    bigquery.SchemaField("total_revenue", "FLOAT64"),
    bigquery.SchemaField("cost_of_revenue", "FLOAT64"),
    bigquery.SchemaField("costof_goods_and_services_sold", "FLOAT64"),
    bigquery.SchemaField("operating_income", "FLOAT64"),
    bigquery.SchemaField("selling_general_and_administrative", "FLOAT64"),
    bigquery.SchemaField("research_and_development", "FLOAT64"),
    bigquery.SchemaField("operating_expenses", "FLOAT64"),
    bigquery.SchemaField("investment_income_net", "FLOAT64"),
    bigquery.SchemaField("net_interest_income", "FLOAT64"),
    bigquery.SchemaField("interest_income", "FLOAT64"),
    bigquery.SchemaField("interest_expense", "FLOAT64"),
    bigquery.SchemaField("non_interest_income", "FLOAT64"),
    bigquery.SchemaField("other_non_operating_income", "FLOAT64"),
    bigquery.SchemaField("depreciation", "FLOAT64"),
    bigquery.SchemaField("depreciation_and_amortization", "FLOAT64"),
    bigquery.SchemaField("income_before_tax", "FLOAT64"),
    bigquery.SchemaField("income_tax_expense", "FLOAT64"),
    bigquery.SchemaField("interest_and_debt_expense", "FLOAT64"),
    bigquery.SchemaField("net_income_from_continuing_operations", "FLOAT64"),
    bigquery.SchemaField("comprehensive_income_net_of_tax", "FLOAT64"),
    bigquery.SchemaField("ebit", "FLOAT64"),
    bigquery.SchemaField("ebitda", "FLOAT64"),
    bigquery.SchemaField("net_income", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    income_statement_df,
    "analyzequicker.analyzequicker.income_statement",
    job_config=bigquery.LoadJobConfig(schema=income_statement_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(income_statement_df)} rows to income_statement table")

#%%
# Combine and load EARNINGS data to BigQuery
earnings_files = glob.glob('data/EARNINGS/*_EARNINGS.csv')
earnings_dfs = []

for file in earnings_files:
    df = pd.read_csv(file)
    earnings_dfs.append(df)

earnings_df = pd.concat(earnings_dfs, ignore_index=True)

# Convert date columns
earnings_df['fiscal_date_ending'] = pd.to_datetime(earnings_df['fiscal_date_ending'], errors='coerce')
earnings_df['reported_date'] = pd.to_datetime(earnings_df['reported_date'], errors='coerce')

# Convert numeric columns
numeric_columns = earnings_df.columns.difference(['fiscal_date_ending', 'reported_date', 'report_time', 'ticker'])
for col in numeric_columns:
    earnings_df[col] = pd.to_numeric(earnings_df[col], errors='coerce')

# Define BigQuery schema for earnings
earnings_schema = [
    bigquery.SchemaField("fiscal_date_ending", "DATE"),
    bigquery.SchemaField("reported_date", "DATE"),
    bigquery.SchemaField("reported_eps", "FLOAT64"),
    bigquery.SchemaField("estimated_eps", "FLOAT64"),
    bigquery.SchemaField("surprise", "FLOAT64"),
    bigquery.SchemaField("surprise_percentage", "FLOAT64"),
    bigquery.SchemaField("report_time", "STRING"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    earnings_df,
    "analyzequicker.analyzequicker.earnings",
    job_config=bigquery.LoadJobConfig(schema=earnings_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(earnings_df)} rows to earnings table")

#%%
# Combine and load DIVIDENDS data to BigQuery
dividends_files = glob.glob('data/DIVIDENDS/*_DIVIDENDS.csv')
dividends_dfs = []

for file in dividends_files:
    df = pd.read_csv(file)
    dividends_dfs.append(df)

dividends_df = pd.concat(dividends_dfs, ignore_index=True)

# Convert date columns
dividends_df['ex_dividend_date'] = pd.to_datetime(dividends_df['ex_dividend_date'], errors='coerce')
dividends_df['declaration_date'] = pd.to_datetime(dividends_df['declaration_date'], errors='coerce')
dividends_df['record_date'] = pd.to_datetime(dividends_df['record_date'], errors='coerce')
dividends_df['payment_date'] = pd.to_datetime(dividends_df['payment_date'], errors='coerce')

# Convert numeric columns
dividends_df['amount'] = pd.to_numeric(dividends_df['amount'], errors='coerce')

# Define BigQuery schema for dividends
dividends_schema = [
    bigquery.SchemaField("ex_dividend_date", "DATE"),
    bigquery.SchemaField("declaration_date", "DATE"),
    bigquery.SchemaField("record_date", "DATE"),
    bigquery.SchemaField("payment_date", "DATE"),
    bigquery.SchemaField("amount", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    dividends_df,
    "analyzequicker.analyzequicker.dividends",
    job_config=bigquery.LoadJobConfig(schema=dividends_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(dividends_df)} rows to dividends table")

#%%
# Combine and load EARNINGS_ESTIMATES data to BigQuery
earnings_estimates_files = glob.glob('data/EARNINGS_ESTIMATES/*_EARNINGS_ESTIMATES.csv')
earnings_estimates_dfs = []

for file in earnings_estimates_files:
    df = pd.read_csv(file)
    earnings_estimates_dfs.append(df)

earnings_estimates_df = pd.concat(earnings_estimates_dfs, ignore_index=True)

# Convert date columns
earnings_estimates_df['date'] = pd.to_datetime(earnings_estimates_df['date'], errors='coerce')

# Convert numeric columns
numeric_columns = earnings_estimates_df.columns.difference(['date', 'horizon', 'ticker'])
for col in numeric_columns:
    earnings_estimates_df[col] = pd.to_numeric(earnings_estimates_df[col], errors='coerce')

# Define BigQuery schema for earnings estimates
earnings_estimates_schema = [
    bigquery.SchemaField("date", "DATE"),
    bigquery.SchemaField("horizon", "STRING"),
    bigquery.SchemaField("eps_estimate_average", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_high", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_low", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_analyst_count", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_average_7_days_ago", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_average_30_days_ago", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_average_60_days_ago", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_average_90_days_ago", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_revision_up_trailing_7_days", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_revision_down_trailing_7_days", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_revision_up_trailing_30_days", "FLOAT64"),
    bigquery.SchemaField("eps_estimate_revision_down_trailing_30_days", "FLOAT64"),
    bigquery.SchemaField("revenue_estimate_average", "FLOAT64"),
    bigquery.SchemaField("revenue_estimate_high", "FLOAT64"),
    bigquery.SchemaField("revenue_estimate_low", "FLOAT64"),
    bigquery.SchemaField("revenue_estimate_analyst_count", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    earnings_estimates_df,
    "analyzequicker.analyzequicker.earnings_estimates",
    job_config=bigquery.LoadJobConfig(schema=earnings_estimates_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(earnings_estimates_df)} rows to earnings_estimates table")

#%%
# Combine and load OVERVIEW data to BigQuery
overview_files = glob.glob('data/OVERVIEW/*_OVERVIEW.csv')
overview_dfs = []

for file in overview_files:
    df = pd.read_csv(file)
    overview_dfs.append(df)

overview_df = pd.concat(overview_dfs, ignore_index=True)

# Convert date columns
overview_df['latest_quarter'] = pd.to_datetime(overview_df['latest_quarter'], errors='coerce')
overview_df['dividend_date'] = pd.to_datetime(overview_df['dividend_date'], errors='coerce')
overview_df['ex_dividend_date'] = pd.to_datetime(overview_df['ex_dividend_date'], errors='coerce')

# Convert numeric columns
numeric_columns = overview_df.columns.difference(['ticker', 'asset_type', 'name', 'description', 'cik', 'exchange', 
                                                   'currency', 'country', 'sector', 'industry', 'address', 
                                                   'official_site', 'fiscal_year_end', 'latest_quarter', 
                                                   'dividend_date', 'ex_dividend_date'])
for col in numeric_columns:
    overview_df[col] = pd.to_numeric(overview_df[col], errors='coerce')

overview_df['cik'] = overview_df['cik'].astype(str)

# Define BigQuery schema for overview
overview_schema = [
    bigquery.SchemaField("ticker", "STRING"),
    bigquery.SchemaField("asset_type", "STRING"),
    bigquery.SchemaField("name", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("cik", "STRING"),
    bigquery.SchemaField("exchange", "STRING"),
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("country", "STRING"),
    bigquery.SchemaField("sector", "STRING"),
    bigquery.SchemaField("industry", "STRING"),
    bigquery.SchemaField("address", "STRING"),
    bigquery.SchemaField("official_site", "STRING"),
    bigquery.SchemaField("fiscal_year_end", "STRING"),
    bigquery.SchemaField("latest_quarter", "DATE"),
    bigquery.SchemaField("market_capitalization", "FLOAT64"),
    bigquery.SchemaField("ebitda", "FLOAT64"),
    bigquery.SchemaField("peratio", "FLOAT64"),
    bigquery.SchemaField("pegratio", "FLOAT64"),
    bigquery.SchemaField("book_value", "FLOAT64"),
    bigquery.SchemaField("dividend_per_share", "FLOAT64"),
    bigquery.SchemaField("dividend_yield", "FLOAT64"),
    bigquery.SchemaField("eps", "FLOAT64"),
    bigquery.SchemaField("revenue_per_share_ttm", "FLOAT64"),
    bigquery.SchemaField("profit_margin", "FLOAT64"),
    bigquery.SchemaField("operating_margin_ttm", "FLOAT64"),
    bigquery.SchemaField("return_on_assets_ttm", "FLOAT64"),
    bigquery.SchemaField("return_on_equity_ttm", "FLOAT64"),
    bigquery.SchemaField("revenue_ttm", "FLOAT64"),
    bigquery.SchemaField("gross_profit_ttm", "FLOAT64"),
    bigquery.SchemaField("diluted_epsttm", "FLOAT64"),
    bigquery.SchemaField("quarterly_earnings_growth_yoy", "FLOAT64"),
    bigquery.SchemaField("quarterly_revenue_growth_yoy", "FLOAT64"),
    bigquery.SchemaField("analyst_target_price", "FLOAT64"),
    bigquery.SchemaField("analyst_rating_strong_buy", "FLOAT64"),
    bigquery.SchemaField("analyst_rating_buy", "FLOAT64"),
    bigquery.SchemaField("analyst_rating_hold", "FLOAT64"),
    bigquery.SchemaField("analyst_rating_sell", "FLOAT64"),
    bigquery.SchemaField("analyst_rating_strong_sell", "FLOAT64"),
    bigquery.SchemaField("trailing_pe", "FLOAT64"),
    bigquery.SchemaField("forward_pe", "FLOAT64"),
    bigquery.SchemaField("price_to_sales_ratio_ttm", "FLOAT64"),
    bigquery.SchemaField("price_to_book_ratio", "FLOAT64"),
    bigquery.SchemaField("evto_revenue", "FLOAT64"),
    bigquery.SchemaField("evto_ebitda", "FLOAT64"),
    bigquery.SchemaField("beta", "FLOAT64"),
    bigquery.SchemaField("_52_week_high", "FLOAT64"),
    bigquery.SchemaField("_52_week_low", "FLOAT64"),
    bigquery.SchemaField("_50_day_moving_average", "FLOAT64"),
    bigquery.SchemaField("_200_day_moving_average", "FLOAT64"),
    bigquery.SchemaField("shares_outstanding", "FLOAT64"),
    bigquery.SchemaField("shares_float", "FLOAT64"),
    bigquery.SchemaField("percent_insiders", "FLOAT64"),
    bigquery.SchemaField("percent_institutions", "FLOAT64"),
    bigquery.SchemaField("dividend_date", "DATE"),
    bigquery.SchemaField("ex_dividend_date", "DATE")
]

job = client.load_table_from_dataframe(
    overview_df,
    "analyzequicker.analyzequicker.overview",
    job_config=bigquery.LoadJobConfig(schema=overview_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(overview_df)} rows to overview table")

#%%
# Combine and load INSIDER_TRANSACTIONS data to BigQuery
insider_transactions_files = glob.glob('data/INSIDER_TRANSACTIONS/*_INSIDER_TRANSACTIONS.csv')
insider_transactions_dfs = []

for file in insider_transactions_files:
    df = pd.read_csv(file)
    insider_transactions_dfs.append(df)

insider_transactions_df = pd.concat(insider_transactions_dfs, ignore_index=True)

# Convert date columns
insider_transactions_df['transaction_date'] = pd.to_datetime(insider_transactions_df['transaction_date'], errors='coerce')

# Convert numeric columns
insider_transactions_df['shares'] = pd.to_numeric(insider_transactions_df['shares'], errors='coerce')
insider_transactions_df['share_price'] = pd.to_numeric(insider_transactions_df['share_price'], errors='coerce')

# Define BigQuery schema for insider transactions
insider_transactions_schema = [
    bigquery.SchemaField("transaction_date", "DATE"),
    bigquery.SchemaField("ticker", "STRING"),
    bigquery.SchemaField("executive", "STRING"),
    bigquery.SchemaField("executive_title", "STRING"),
    bigquery.SchemaField("security_type", "STRING"),
    bigquery.SchemaField("acquisition_or_disposal", "STRING"),
    bigquery.SchemaField("shares", "FLOAT64"),
    bigquery.SchemaField("share_price", "FLOAT64")
]

job = client.load_table_from_dataframe(
    insider_transactions_df,
    "analyzequicker.analyzequicker.insider_transactions",
    job_config=bigquery.LoadJobConfig(schema=insider_transactions_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(insider_transactions_df)} rows to insider_transactions table")

#%%
# Combine and load SHARES_OUTSTANDING data to BigQuery
shares_outstanding_files = glob.glob('data/SHARES_OUTSTANDING/*_SHARES_OUTSTANDING.csv')
shares_outstanding_dfs = []

for file in shares_outstanding_files:
    df = pd.read_csv(file)
    shares_outstanding_dfs.append(df)

shares_outstanding_df = pd.concat(shares_outstanding_dfs, ignore_index=True)

# Convert date columns
shares_outstanding_df['date'] = pd.to_datetime(shares_outstanding_df['date'], errors='coerce')

# Convert numeric columns
shares_outstanding_df['shares_outstanding_diluted'] = pd.to_numeric(shares_outstanding_df['shares_outstanding_diluted'], errors='coerce')
shares_outstanding_df['shares_outstanding_basic'] = pd.to_numeric(shares_outstanding_df['shares_outstanding_basic'], errors='coerce')

# Define BigQuery schema for shares outstanding
shares_outstanding_schema = [
    bigquery.SchemaField("date", "DATE"),
    bigquery.SchemaField("shares_outstanding_diluted", "FLOAT64"),
    bigquery.SchemaField("shares_outstanding_basic", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    shares_outstanding_df,
    "analyzequicker.analyzequicker.shares_outstanding",
    job_config=bigquery.LoadJobConfig(schema=shares_outstanding_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(shares_outstanding_df)} rows to shares_outstanding table")

#%%
# Combine and load ETF_HOLDINGS data to BigQuery
etf_holdings_files = glob.glob('data/ETF_PROFILE/ETF_HOLDINGS/*_ETF_HOLDINGS.csv')
etf_holdings_dfs = []

for file in etf_holdings_files:
    df = pd.read_csv(file)
    etf_holdings_dfs.append(df)

etf_holdings_df = pd.concat(etf_holdings_dfs, ignore_index=True)

# Convert numeric columns
etf_holdings_df['weight'] = pd.to_numeric(etf_holdings_df['weight'], errors='coerce')

# Define BigQuery schema for ETF holdings
etf_holdings_schema = [
    bigquery.SchemaField("symbol", "STRING"),
    bigquery.SchemaField("description", "STRING"),
    bigquery.SchemaField("weight", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    etf_holdings_df,
    "analyzequicker.analyzequicker.etf_holdings",
    job_config=bigquery.LoadJobConfig(schema=etf_holdings_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(etf_holdings_df)} rows to etf_holdings table")

#%%
# Combine and load ETF_INFO data to BigQuery
etf_info_files = glob.glob('data/ETF_PROFILE/ETF_INFO/*_ETF_INFO.csv')
etf_info_dfs = []

for file in etf_info_files:
    df = pd.read_csv(file)
    etf_info_dfs.append(df)

etf_info_df = pd.concat(etf_info_dfs, ignore_index=True)

# Convert date columns
etf_info_df['inception_date'] = pd.to_datetime(etf_info_df['inception_date'], errors='coerce')

# Convert numeric columns
numeric_columns = etf_info_df.columns.difference(['inception_date', 'leveraged', 'ticker'])
for col in numeric_columns:
    etf_info_df[col] = pd.to_numeric(etf_info_df[col], errors='coerce')

# Define BigQuery schema for ETF info
etf_info_schema = [
    bigquery.SchemaField("net_assets", "FLOAT64"),
    bigquery.SchemaField("net_expense_ratio", "FLOAT64"),
    bigquery.SchemaField("portfolio_turnover", "FLOAT64"),
    bigquery.SchemaField("dividend_yield", "FLOAT64"),
    bigquery.SchemaField("inception_date", "DATE"),
    bigquery.SchemaField("leveraged", "STRING"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    etf_info_df,
    "analyzequicker.analyzequicker.etf_info",
    job_config=bigquery.LoadJobConfig(schema=etf_info_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(etf_info_df)} rows to etf_info table")

#%%
# Combine and load ETF_SECTORS data to BigQuery
etf_sectors_files = glob.glob('data/ETF_PROFILE/ETF_SECTORS/*_ETF_SECTORS.csv')
etf_sectors_dfs = []

for file in etf_sectors_files:
    df = pd.read_csv(file)
    etf_sectors_dfs.append(df)

etf_sectors_df = pd.concat(etf_sectors_dfs, ignore_index=True)

# Convert numeric columns
etf_sectors_df['weight'] = pd.to_numeric(etf_sectors_df['weight'], errors='coerce')

# Define BigQuery schema for ETF sectors
etf_sectors_schema = [
    bigquery.SchemaField("sector", "STRING"),
    bigquery.SchemaField("weight", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    etf_sectors_df,
    "analyzequicker.analyzequicker.etf_sectors",
    job_config=bigquery.LoadJobConfig(schema=etf_sectors_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(etf_sectors_df)} rows to etf_sectors table")

#%%
# Combine and load SPLITS data to BigQuery
splits_files = glob.glob('data/SPLITS/*_SPLITS.csv')
splits_dfs = []

for file in splits_files:
    df = pd.read_csv(file)
    splits_dfs.append(df)

splits_df = pd.concat(splits_dfs, ignore_index=True)

# Convert date columns
splits_df['effective_date'] = pd.to_datetime(splits_df['effective_date'], errors='coerce')

# Convert numeric columns
splits_df['split_factor'] = pd.to_numeric(splits_df['split_factor'], errors='coerce')

# Define BigQuery schema for splits
splits_schema = [
    bigquery.SchemaField("effective_date", "DATE"),
    bigquery.SchemaField("split_factor", "FLOAT64"),
    bigquery.SchemaField("ticker", "STRING")
]

job = client.load_table_from_dataframe(
    splits_df,
    "analyzequicker.analyzequicker.splits",
    job_config=bigquery.LoadJobConfig(schema=splits_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(splits_df)} rows to splits table")

#%%
# Combine and load COMMODITIES data to BigQuery
commodities_files = glob.glob('data/COMMODITIES/*.csv')
commodities_dfs = []

for file in commodities_files:
    df = pd.read_csv(file)
    commodities_dfs.append(df)

commodities_df = pd.concat(commodities_dfs, ignore_index=True)

# Convert date columns
commodities_df['timestamp'] = pd.to_datetime(commodities_df['timestamp'], errors='coerce')

# Convert numeric columns
commodities_df['value'] = pd.to_numeric(commodities_df['value'], errors='coerce')

# Define BigQuery schema for commodities
commodities_schema = [
    bigquery.SchemaField("timestamp", "DATE"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("commodity", "STRING")
]

job = client.load_table_from_dataframe(
    commodities_df,
    "analyzequicker.analyzequicker.commodities",
    job_config=bigquery.LoadJobConfig(schema=commodities_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(commodities_df)} rows to commodities table")

#%%
# Combine and load ECONOMIC_INDICATORS data to BigQuery
economic_indicators_files = glob.glob('data/ECONOMIC_INDICATORS/*.csv')
economic_indicators_dfs = []

for file in economic_indicators_files:
    df = pd.read_csv(file)
    economic_indicators_dfs.append(df)

economic_indicators_df = pd.concat(economic_indicators_dfs, ignore_index=True)

# Convert date columns
economic_indicators_df['timestamp'] = pd.to_datetime(economic_indicators_df['timestamp'], errors='coerce')

# Convert numeric columns
economic_indicators_df['value'] = pd.to_numeric(economic_indicators_df['value'], errors='coerce')

# Define BigQuery schema for economic indicators
economic_indicators_schema = [
    bigquery.SchemaField("timestamp", "DATE"),
    bigquery.SchemaField("value", "FLOAT64"),
    bigquery.SchemaField("economic_indicator", "STRING")
]

job = client.load_table_from_dataframe(
    economic_indicators_df,
    "analyzequicker.analyzequicker.economic_indicators",
    job_config=bigquery.LoadJobConfig(schema=economic_indicators_schema, write_disposition="WRITE_TRUNCATE")
)
job.result()
print(f"Loaded {len(economic_indicators_df)} rows to economic_indicators table")

#%%
