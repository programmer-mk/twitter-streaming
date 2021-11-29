import pandas as pd
import numpy as np

base_dir = '/Users/mkovacevic/Desktop/DataScience/twitter-streaming/aws/cloudformation/stock-price-data/company-historical-stock-prices'

microsoft_data = pd.read_csv(f'{base_dir}/historical/MSFT.US-historical-stocks.csv')
amazon_data = pd.read_csv(f'{base_dir}/historical/AMZN.US-historical-stocks.csv')
apple_data = pd.read_csv(f'{base_dir}/historical/AAPL.US-historical-stocks.csv')
tesla_data = pd.read_csv(f'{base_dir}/historical/TSLA.US-historical-stocks.csv')
bitcoin_data = pd.read_csv(f'{base_dir}/historical/BTC-USD.CC-historical-stocks.csv')

if __name__ == '__main__':
    print("Start processing")

    microsoft_data['Polarity'] = np.random.randint(0.1, 6.5, microsoft_data.shape[0]).astype(float)
    microsoft_data.to_csv(f'{base_dir}/mocked-data-with-polarity/MSFT.US-with-polarity.csv', index=False)

    amazon_data['Polarity'] = np.random.randint(0.1, 6.5, amazon_data.shape[0]).astype(float)
    amazon_data.to_csv(f'{base_dir}/mocked-data-with-polarity/AMZN.US-with-polarity.csv', index=False)

    apple_data['Polarity'] = np.random.randint(0.1, 6.5, apple_data.shape[0]).astype(float)
    apple_data.to_csv(f'{base_dir}/mocked-data-with-polarity/AAPL.US-with-polarity.csv', index=False)

    tesla_data['Polarity'] = np.random.randint(0.1, 6.5, tesla_data.shape[0]).astype(float)
    tesla_data.to_csv(f'{base_dir}/mocked-data-with-polarity/TSLA.US-with-polarity.csv', index=False)

    bitcoin_data['Polarity'] = np.random.randint(0.1, 6.5, bitcoin_data.shape[0]).astype(float)
    bitcoin_data.to_csv(f'{base_dir}/mocked-data-with-polarity/BTC-USD.CC-with-polarity.csv', index=False)

    print("Processing done")