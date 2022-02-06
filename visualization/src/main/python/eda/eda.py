import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime

nltk.download('vader_lexicon')
nltk.download('punkt')
stop_words=stopwords.words('english')
sia=SentimentIntensityAnalyzer()


def create_test_dataset(data):
    return pd.concat([data.head(10001), data.loc[200000:2010000], data.loc[2000000:2001000], data.tail(10000)], ignore_index=True)


def remove_stopwords(txt):
    words=word_tokenize(txt)
    words=[word for word in words if word.isalpha()]
    words=[word for word in words if not word in stop_words]
    return ' '.join(words)

def transform_with_sentiment(tweets):
    tweet_polarity=tweets["body"].apply(lambda z:sia.polarity_scores((z)))
    print(tweet_polarity)
    positive=[]
    negative=[]
    neutral=[]
    sentiment=[]
    compound=[]

    for ent in tweet_polarity:
        #positive.append(ent['pos'])
        #negative.append(ent['neg'])
        #neutral.append(ent['neu'])
        compound.append(ent["compound"])
        if (ent['compound'] > 0.05):
            positive.append(1.0)
            negative.append(0.0)
            neutral.append(0.0)
            sentiment.append("positive")
        elif (ent['compound'] < -0.05):
            sentiment.append("negative")
            positive.append(0.0)
            negative.append(1.0)
            neutral.append(0.0)
        else:
            positive.append(0.0)
            negative.append(0.0)
            neutral.append(1.0)
            sentiment.append("neutral")

    sentiment_scores_columns = pd.DataFrame({"positive":positive,"negative":negative,"neutral":neutral,"compound":compound,"sentiment":sentiment})
    merged_data = pd.concat([tweets, sentiment_scores_columns], axis=1)
    merged_data.head(10)
    return merged_data


def company_tweet_percent(data):
    data.company_name.value_counts().plot(kind="pie",autopct="%1.0f%%").axis('off')
    print('stop execution here.')


def tweet_percent_per_year(data):
    data.year.value_counts().plot(kind="pie",autopct="%1.0f%%").axis('off')
    print('stop execution here.')


def sentiment_group_percent_per_company(data):
    x, y = "company_name","sentiment"
    grouped_data = data.groupby(x)[y] \
        .value_counts(normalize=True) \
        .mul(100) \
        .rename('percent') \
        .reset_index()

    sns.catplot(x=x,y='percent',hue=y,kind='bar',data=grouped_data).set(xlabel=None)
    print('stop execution here.')


def companies_trolled(data):
    company=[]
    author=data.groupby(["writer"]).sum().nsmallest(10,"compound").reset_index().writer
    print(author)
    for w in author:
        company.append(data.loc[data['writer'] == w, 'company_name'].iloc[0])
    troll=pd.DataFrame(list(zip(author,company)),columns=['author','company'])
    sns.countplot(x="company",data=troll).set_title("Companies trolled by authors").axis('off')
    print('stop execution here.')


def transform_date_column(data, column_name):
    data[column_name] = data[column_name].apply(lambda timestamp: datetime.fromtimestamp(timestamp))
    data[column_name] = pd.to_datetime(data[column_name])
    data["day_date"] = data[column_name].apply(lambda date: date.strftime("%Y-%m-%d"))
    return data.drop(column_name, axis=1)


def load_stock_data():
    stock_values = pd.read_csv('CompanyValues.csv')
    stock_values['fluctuation'] = stock_values.high_value - stock_values.low_value
    stock_values['price_gain'] = stock_values.close_value - stock_values.open_value
    stock_values['total_valuation_EOD'] = stock_values.volume * stock_values.close_value
    return stock_values


def draw_stock_price_with_sentiment(merged_data, score_name="compound"):
    ticker_symbol = 'MSFT'
    sentiment_overtime(merged_data[merged_data['ticker_symbol'] == ticker_symbol], ticker_symbol, score_column_name=score_name)


def sentiment_overtime(merged_data, title, score_column_name="compound"):
    print("\n\n")
    fig = plt.figure(figsize=(24,10))
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    ax1.vlines(merged_data['day_date'], 0, merged_data[score_column_name])
    ax1.axhline(y=0, color='r', linestyle='-')
    ax2.plot(merged_data['day_date'], merged_data['close_value'], color='orange', label='Stock price')
    ax2.set_title("Effects of " + title +" tweets to stock price")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.set_xlabel('Day date')
    ax1.set_xticks(ticks=[10.0, 210.0, 420.0, 630.0, 840.0])
    ax1.set_xticklabels(['2015', '2016', '2017', '2018', '2019'])
    ax1.set_ylabel('Sentiment aggregated score', color="blue")

    ax2.legend(lines + lines2, labels + labels2, loc=0)
    plt.show()


def compute_days_passed(data):
    data['start_date'] = '2015-01-01'
    start_dt = pd.to_datetime(data.start_date.values, format="%Y-%m-%d")
    dt = pd.to_datetime(data.day_date.values, format="%Y-%m-%d")

    diff = (dt - start_dt).days
    data["day_passed"] = diff.tolist()

    data.drop('start_date', axis=1, inplace=True)
    return data


"""
    data: whole tweets dataset
    company_name: can be in group (Microsoft, Google Inc, Tesla Inc, Amazon.com, apple
    sentiment: negative or positive
"""
def tweets_count_from_beginning(data, company_name, sentiment):
    filter_data = data[data['company_name'] == company_name]
    sentiment_agg_day_passed = filter_data[["day_passed",sentiment]].groupby(["day_passed"]).sum()
    plt.scatter(sentiment_agg_day_passed.index, sentiment_agg_day_passed[sentiment].values)
    plt.xlabel("Days after Jan. 1, 2015", fontsize=12)
    plt.ylabel(f"{sentiment} tweets", fontsize=12)
    plt.title(company_name, fontsize=14)
    plt.show()


"""
    data: whole tweets dataset
    company_name: can be in group (Microsoft, Google Inc, Tesla Inc, Amazon.com, apple
"""
def close_value_from_beginning(data, company_sym):
    filter_data = data[data['ticker_symbol'] == company_sym]
    market_val_agg_day_passed = filter_data[["day_passed", "close_value"]].groupby(["day_passed"]).sum()
    plt.scatter(market_val_agg_day_passed.index, market_val_agg_day_passed["close_value"].values)
    plt.xlabel("Days after Jan. 1, 2015", fontsize=12)
    plt.ylabel(f"Market value", fontsize=12)
    plt.title(company_sym, fontsize=14)
    plt.show()


if __name__ == '__main__':
    reuse = True
    if not reuse:
        print('Start loading data...')
        company = pd.read_csv('Company.csv')
        company_tweet = pd.read_csv('Company_Tweet.csv')
        tweet = pd.read_csv('Tweet.csv')
        stock_values = load_stock_data()
        print('Data loaded!')

        # merge datasets
        name_tick_dic=pd.Series(company.company_name.values, index=company.ticker_symbol).to_dict()
        company_tweet.ticker_symbol=company_tweet.ticker_symbol.map(name_tick_dic)
        tick_id_dict=pd.Series(company_tweet.ticker_symbol.values,index=company_tweet.tweet_id).to_dict()
        tweet.tweet_id=tweet.tweet_id.map(tick_id_dict)
        tweet.rename(columns={"tweet_id":"company_name"},inplace=True)

        # tweet.body = tweet["body"].apply(remove_stopwords)
        # print("Stop words removed")
        extended_data = transform_with_sentiment(tweet)
        print("Sentiments added")
        extended_data.to_csv("Tweet_Processed.csv", index=False)
        print("Saved processed tweets to disk")
    else:
        extended_data = pd.read_csv('Tweet_Processed.csv')
        stock_values = load_stock_data()

    extended_data = transform_date_column(extended_data, "post_date")
    extended_data = compute_days_passed(extended_data)
    stock_values = compute_days_passed(stock_values)


    # 1. company_tweet_percent(extended_data)

    # 2. tweet_percent_per_year(extended_data)

    # 3. sentiment_group_percent_per_company(extended_data)

    # 4. companies_trolled(extended_data)


    compund_agg = extended_data[["day_date", "compound"]].groupby(["day_date"]).sum()["compound"].to_frame()
    merged_data = pd.merge(compund_agg, stock_values, on="day_date")

    # 5.draw_stock_price_with_sentiment(merged_data)

    # 6. tweets_count_from_beginning(extended_data, "Microsoft", "negative")
    # 6. tweets_count_from_beginning(extended_data, "Tesla Inc", "negative")
    # 6. tweets_count_from_beginning(extended_data, "Amazon.com", "negative")
    # 6. tweets_count_from_beginning(extended_data, "apple", "negative")
    # 6. tweets_count_from_beginning(extended_data, "Google Inc", "negative")

    # 6. tweets_count_from_beginning(extended_data, "Microsoft", "positive")
    # 6. tweets_count_from_beginning(extended_data, "Tesla Inc", "positive")
    # 6. tweets_count_from_beginning(extended_data, "Amazon.com", "positive")
    # 6. tweets_count_from_beginning(extended_data, "apple", "positive")
    # 6. tweets_count_from_beginning(extended_data, "Google Inc", "positive")

    """
       6. close_value_from_beginning(stock_values, "GOOGL")
    
       6. close_value_from_beginning(stock_values, "TSLA")
    
       6. close_value_from_beginning(stock_values, "AMZN")
          
       6. close_value_from_beginning(stock_values, "MSFT")
          
       6. close_value_from_beginning(stock_values, "APPL")
        
    """

    tesla_df = stock_values[stock_values['ticker_symbol'] == 'TSLA']
    tesla_df["delta_close_value"] = np.array([0] + list(tesla_df.close_value.values[1:] - tesla_df.close_value.values[:-1]))

    print('')


