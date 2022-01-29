import pandas as pd

#//tweet_id,writer,post_date,body,comment_num,retweet_num,like_num

def basic_text_preprocessing(input):
    return input.replace(',', '').replace('"', '')

def main():
    data = pd.read_csv('./Tweet.csv')
    data['body'] = data['body'].apply(basic_text_preprocessing)
    data.to_csv('./Tweet_Preprocessed.csv',index=False)

if __name__ == '__main__':
    main()