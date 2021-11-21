import urllib3

target_company_symbols = ['TSLA.US','MSFT.US', 'AMZN.US', 'AAPL.US', 'BTC-USD.CC']

def get_stock_daily_data(symbol):
    http = urllib3.PoolManager()
    params = {
       "api_token": "OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX" # this is API test token
    }
    url = f'https://eodhistoricaldata.com/api/eod/{symbol}'
    try:
        print(f'url : {url}')
        response = http.request("GET", url, fields=params)
        data = str(response.data, 'utf-8')

        text_file = open(f"company-historical-stock-prices/{symbol}-historical-stocks.csv", "w")
        text_file.write(data)
        text_file.close()

    except Exception as e:
        print('Exception occured: {}. Stock data was not fetched'.format(e))


import pandas as pd
import boto3
import os

os.environ['BUCKET_NAME'] = "twitter-analysis-platform-bucket"
os.environ['S3_KEY'] = 'xxx'
os.environ['TWEETS_AGGREGATED_POLARITY_KEY_PREFIX'] = "tweet-aggregated-results"
os.environ['STOCK_PRICE_KEY_PREFIX'] = "stock-price-data"
os.environ['MERGED_DATA_KEY_PREFIX'] = "merged-data"
os.environ['TRAINING_CHUNK_PERCENT'] = "80.0"

s3_client = boto3.client('s3')
bucket_name = os.environ['BUCKET_NAME']

# ALL-SYMBOLS-POLARITY
def pull_computed_polarity():
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name)
    tweets_aggregated_s3_key = os.environ['TWEETS_AGGREGATED_POLARITY_KEY_PREFIX']
    df_s3_data = None
    for object_summary in my_bucket.objects.filter(Prefix=tweets_aggregated_s3_key):
        if object_summary.key.endswith('.csv'):
            response = s3_client.get_object(Bucket=bucket_name, Key=object_summary.key)
            df_s3_data = pd.read_csv(response['Body'], sep=',').dropna()
    return df_s3_data


def merge_symbol_data_with_polarity(symbol_name, polarity_data, search_term_dict):
    s3_key = f'{os.environ["STOCK_PRICE_KEY_PREFIX"]}/{symbol_name}-daily-stocks.csv'
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    stock_data = pd.read_csv(response['Body'], sep=',').dropna()
    symbol_polarity = polarity_data[polarity_data['search_term'] == search_term_dict[symbol_name]]
    #symbol_polarity.rename(columns={"date": "Date"})
    joined_data = stock_data.merge(symbol_polarity, left_on='Date', right_on='date')
    #joined_data = stock_data.join(symbol_polarity, how="inner", on="Date")
    print(joined_data.head())
    return joined_data


def split_data(data, symbol_name):
    training_s3_key = f'{os.environ["MERGED_DATA_KEY_PREFIX"]}/training/{symbol_name}-with-polarity.csv'
    testing_s3_key = f'{os.environ["MERGED_DATA_KEY_PREFIX"]}/testing/{symbol_name}-with-polarity.csv'
    print(f'start writing {symbol_name} data to s3...')
    from io import StringIO
    import numpy as np

    last_training_index = int((float(os.environ['TRAINING_CHUNK_PERCENT']) / 100.00) * data.shape[0])

    training_data = data.iloc[:last_training_index]
    testing_data = data.iloc[last_training_index:]

    csv_buffer_training = StringIO()
    training_data.to_csv(csv_buffer_training)

    csv_buffer_testing = StringIO()
    testing_data.to_csv(csv_buffer_testing)

    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, f'{training_s3_key}').put(Body=csv_buffer_training.getvalue())
    s3_resource.Object(bucket_name, f'{testing_s3_key}').put(Body=csv_buffer_testing.getvalue())


if __name__ == '__main__':
    # for symbol in target_company_symbols:
    #     get_stock_daily_data(symbol)
    polarity_data = pull_computed_polarity()
    dict = {
        'TSLA.US': 'tesla',
        'MSFT.US': 'microsoft',
        'AMZN.US': 'amazon',
        'AAPL.US': 'apple',
        'BTC-USD.CC': 'bitcoin'
    }
    target_company_symbols = ['TSLA.US','MSFT.US', 'AMZN.US', 'AAPL.US', 'BTC-USD.CC']
    for symbol in target_company_symbols:
        """
            merged data contains stock data + aggregated polarity of tweets on that day
        """
        merged_data = merge_symbol_data_with_polarity(symbol, polarity_data, dict)
        split_data(merged_data, symbol)