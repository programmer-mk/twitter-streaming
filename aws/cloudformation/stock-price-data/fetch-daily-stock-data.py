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

if __name__ == '__main__':
    for symbol in target_company_symbols:
        get_stock_daily_data(symbol)