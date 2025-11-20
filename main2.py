#%%
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
        logging.error('Request failed')

    if datatype == 'csv':
        data = r.content.decode('utf-8')
        actual_columns = data.splitlines()[0].split(',')
        if len(actual_columns) <= 1:
            logging.error(f'Data is not in the expected format - {symbol} - {function}')
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
            raise Exception(f'Data is not in the expected format - {symbol} - {function}'.format(symbol, function))

    df.columns = df.columns.str.replace(r'([a-z0-9])([A-Z])', r'\1_\2', regex=True).str.lower()
    df.columns = df.columns.str.replace(r'^(\d)', r'_\1', regex=True)
    
    # if numeric column, replace None with 0
    for col in df.columns:
        if col != 'symbol':
            temp = pd.to_numeric(df[col], errors='coerce')
            # If at least some values can be converted to numeric, use the conversion
            if not temp.isna().all():
                df[col] = df[col].replace('None', 0)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

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
import logging
from datetime import datetime

os.makedirs('logs', exist_ok=True)
log_time = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
logging.basicConfig(filename=f'logs/{log_time} main2.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
df.to_csv('data/LISTING_STATUS.csv', index=False)

#%%
stock_symbols = df[df['asset_type'] == 'Stock']['symbol'].unique()

#%%
for symbol in tqdm.tqdm(stock_symbols):
    try:
        call_alpha_vantage(symbol, {'function': 'OVERVIEW'})
        call_alpha_vantage(symbol, {'function': 'TIME_SERIES_DAILY_ADJUSTED', 'datatype': 'csv'})
    except Exception as e:
        logging.error(f'Error for {symbol}: {e}')

#%%
from google.cloud import bigquery
from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file(config.service_account_path)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)
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
import glob

#%%
# Combine and load OVERVIEW data to BigQuery
overview_files = glob.glob('data/OVERVIEW/*_OVERVIEW.csv')
overview_dfs = []

for file in overview_files:
    df = pd.read_csv(file)
    overview_dfs.append(df)

overview_df = pd.concat(overview_dfs, ignore_index=True)
# overview_df.columns = overview_df.columns.str.replace(r'^(\d)', r'_\1', regex=True)

# Convert date columns
overview_df['latest_quarter'] = pd.to_datetime(overview_df['latest_quarter'], errors='coerce')
overview_df['dividend_date'] = pd.to_datetime(overview_df['dividend_date'], errors='coerce')
overview_df['ex_dividend_date'] = pd.to_datetime(overview_df['ex_dividend_date'], errors='coerce')

# Convert numeric columns
numeric_columns = overview_df.columns.difference(['symbol', 'asset_type', 'name', 'description', 'cik', 'exchange', 
                                                   'currency', 'country', 'sector', 'industry', 'address', 
                                                   'official_site', 'fiscal_year_end', 'latest_quarter', 
                                                   'dividend_date', 'ex_dividend_date'])
for col in numeric_columns:
    overview_df[col] = pd.to_numeric(overview_df[col], errors='coerce')

overview_df['symbol'] = overview_df['symbol'].astype(str)
overview_df['cik'] = overview_df['cik'].astype(str)

# Define BigQuery schema for overview
overview_schema = [
    bigquery.SchemaField("symbol", "STRING"),
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
# Combine and load TIME_SERIES_DAILY_ADJUSTED data to BigQuery
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
    bigquery.SchemaField("symbol", "STRING")
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

