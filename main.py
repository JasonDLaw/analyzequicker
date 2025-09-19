#%%
import pandas as pd
import time
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv
import os

load_dotenv()
service_account_path = os.getenv('service_account_path')
credentials = service_account.Credentials.from_service_account_file(service_account_path)
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

#%%
# read in tickers
# https://www.nasdaq.com/market-activity/stocks/screener
tickers = pd.read_csv('data/nasdaq_nyse_amex.csv')
tickers['Symbol'] = tickers['Symbol'].str.replace('/', '-')

# %%
