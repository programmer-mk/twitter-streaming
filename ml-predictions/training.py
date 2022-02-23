import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate, KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import boto3
import os
import time
import datetime
import matplotlib.pyplot as plt

s3_client = boto3.client('s3')
# bucket_name = os.environ['BUCKET_NAME']
# merged_data_key_prefix = os.environ["MERGED_DATA_KEY_PREFIX"]

companies = {
    "AAPL" : "apple",
    "GOOG" : "Google Inc",
    "AMZN" : "Amazon.com",
    "TSLA" : "Tesla Inc",
    "MSFT":"Microsoft"
}

def read_data(all_stocks=True):
    if not all_stocks:
        microsoft_data = pd.read_csv('data/polarity/MSFT.US-with-polarity.csv')
        amazon_data = pd.read_csv('data/polarity/AMZN.US-with-polarity.csv')
        apple_data = pd.read_csv('data/polarity/AAPL.US-with-polarity.csv')
        tesla_data = pd.read_csv('data/polarity/TSLA.US-with-polarity.csv')
        bitcoin_data = pd.read_csv('data/polarity/GOOG.US-with-polarity.csv')
        return [[microsoft_data, "MSFT"], [amazon_data, "AMZN"], [apple_data, "AAPL"], [tesla_data, "TSLA"], [bitcoin_data, "GOOG"]]
    else:
        return pd.read_csv('data/polarity/all-data-with-polarity.csv')

def SMA(data, period=30, column='close_value'):
    return data[column].rolling(window=period).mean()


def EMA(data, period=20, column='close_value'):
    return data[column].ewm(span=period, adjust=False).mean()


def MACD(data, period_long=26, period_short=12, period_signal=9, column='close_value'):
    short_ema = EMA(data, period_short, column=column)
    long_ema = EMA(data, period_long, column=column)
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = EMA(data, period_signal, column='MACD')
    return data


def RSI(data, period=14, column='close_value'):
    delta = data[column].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    data['up'] = up
    data['down'] = down
    AVG_Gain = SMA(data, period, column='up')
    AVG_Loss = abs(SMA(data, period, column='down'))
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0/ (1.0 + RS))

    data['RSI'] = RSI
    return data


def save_models(estimators):
    for estimator in estimators:
        for estimator_key in estimator:
            if estimator_key != "stock_name":
                joblib.dump(estimator[estimator_key], f'/models/{estimator_key}-{estimator["stock_name"]}-estimator.pkl', compress=1)


def choose_target(x):
    diff = x.iloc[-1] - x.iloc[0]
    if diff > 0:
        return 1
    else:
        return 0

def compute_target_column(data, days_after):
    rolling_closes = data['close_value'].rolling(window=(days_after+1)).apply(choose_target).replace(np.nan, 0) # if before x days price is lower than todays price put 1 else put 0
    data.insert(3, "Target", rolling_closes)
    return data

def preprocess_data(data, days_after=7):
    #MACD(data)
    #RSI(data)
    #data['SMA'] = SMA(data)
    #data['EMA'] = EMA(data)
    data = data.dropna()
    #Create the target column
    data = data.sort_values(by=['date'])

    data_build = data[(data['date'] >= '2015-01-01') & (data['date'] < '2020-01-01')]
    data_build = compute_target_column(data_build, days_after)
    return data_build
    # features = data_build[keep_columns].values
    # labels = data_build['Target'].values
    # return features, labels


def nested_cv_logistic_regression(X, Y, stock_name):
    param_grid = {"logisticregression__C": [0.0001, 0.0005, 0.001, 0.01, 0.3, 0.5, 1, 5, 10],
              "logisticregression__penalty":["l1","l2"],
              "logisticregression__solver": ["liblinear", "saga"]
              }

    lgr = LogisticRegression(max_iter=20000)
    pipeline = Pipeline([('scaler', StandardScaler()), ('logisticregression', lgr)])

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Nested CV with parameter optimization
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv)
    nested_scores = cross_validate(clf, X=X, y=Y, scoring=['accuracy', 'f1', 'precision', 'recall'], cv=outer_cv)
    accuracy_score = nested_scores['test_accuracy'].mean()
    f1_score = nested_scores['test_f1'].mean()
    precision_score = nested_scores['test_precision'].mean()
    recall_score = nested_scores['test_recall'].mean()

    print(f'#{stock_name} Results after cross-validation logistic regression: \n')
    print(f'#{stock_name} accuracy: {accuracy_score}, f1: {f1_score}, precision: {precision_score}, recall: {recall_score}  \n')

    clf.fit(X, Y)
    print(f'#{stock_name} best estimator: {clf.best_estimator_}')
    print(f'#{stock_name} best params: {clf.best_params_}')
    return clf, accuracy_score


def nested_cv_svm(X, Y, stock_name):
    param_grid = {'svm__C': [0.0001, 0.0005, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
                  'svm__gamma': [100, 10, 1, 0.1, 0.01, 0.001],
                  'svm__kernel': ['rbf']}

    svm = SVC()
    pipeline = Pipeline([('scaler', StandardScaler()), ('svm', svm)])

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Nested CV with parameter optimization
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
    nested_scores = cross_validate(clf, X=X, y=Y, scoring=['accuracy', 'f1', 'precision', 'recall'], cv=outer_cv)
    print('dt scores completation...')
    accuracy_score = nested_scores['test_accuracy'].mean()
    f1_score = nested_scores['test_f1'].mean()
    precision_score = nested_scores['test_precision'].mean()
    recall_score = nested_scores['test_recall'].mean()

    print(f'#{stock_name} Results after cross-validation support vector machine: \n')
    print(f'#{stock_name} accuracy: {accuracy_score}, f1: {f1_score}, precision: {precision_score}, recall: {recall_score}  \n')

    clf.fit(X, Y)
    print(f'#{stock_name} best estimator: {clf.best_estimator_}')
    print(f'#{stock_name} best params: {clf.best_params_}')

    return clf, accuracy_score


def nested_cv_decission_tree(X, Y, stock_name):
    param_grid = {'tree__max_features': ['auto', 'sqrt', 'log2'],
                  'tree__ccp_alpha': [0.1, .01, .001],
                  'tree__max_depth' : [5, 6, 7, 8, 9, 12, 15, 18],
                  'tree__criterion' :['gini', 'entropy']
                  }

    tree_clas = DecisionTreeClassifier(random_state=1024)
    pipeline = Pipeline([('scaler', StandardScaler()), ('tree', tree_clas)])
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Nested CV with parameter optimization
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
    nested_scores = cross_validate(clf, X=X, y=Y, scoring=['accuracy', 'f1', 'precision', 'recall'], cv=outer_cv)
    accuracy_score = nested_scores['test_accuracy'].mean()
    f1_score = nested_scores['test_f1'].mean()
    precision_score = nested_scores['test_precision'].mean()
    recall_score = nested_scores['test_recall'].mean()

    print(f'#{stock_name} Results after cross-validation decission tree: \n')
    print(f'#{stock_name} accuracy: {accuracy_score}, f1: {f1_score}, precision: {precision_score}, recall: {recall_score}  \n')

    clf.fit(X, Y)
    print(f'#{stock_name} best estimator: {clf.best_estimator_}')
    print(f'#{stock_name} best params: {clf.best_params_}')

    return clf, accuracy_score


def download_data():
    print("Downloading data started...")
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket_name)
    full_prefix = f'{merged_data_key_prefix}/training'
    for object_summary in my_bucket.objects.filter(Prefix=full_prefix):
        if object_summary.key.endswith('.csv'):
            response = s3_client.get_object(Bucket=bucket_name, Key=object_summary.key)
            df_s3_data = pd.read_csv(response['Body'], sep=',')
            print(f'downloaded file: {object_summary.key.lstrip(full_prefix)}')
            df_s3_data.to_csv(f'/data/{object_summary.key.lstrip(full_prefix)}', sep=',', index=False)


def package_training_data(merge_stocks=True):

    extended_data = pd.read_csv('data/Spark_Tweet_Output.csv')
    stock_values = pd.read_csv('data/CompanyValues.csv')
    merged_data = pd.merge(extended_data, stock_values, how="inner", left_on=["date", "search_term"], right_on=["day_date", "ticker_symbol"])
    modeling_columns = ["date", "agg_polarity", "close_value", "volume", "open_value", "high_value", "low_value", "search_term"]

    if merge_stocks:
        merged_data[modeling_columns].to_csv('data/polarity/all-data-with-polarity.csv', index=False)

    merged_data[merged_data["search_term"] == "AAPL"][modeling_columns].to_csv('data/polarity/AAPL.US-with-polarity.csv', index=False)
    merged_data[merged_data["search_term"] == "MSFT"][modeling_columns].to_csv('data/polarity/MSFT.US-with-polarity.csv', index=False)
    merged_data[merged_data["search_term"] == "AMZN"][modeling_columns].to_csv('data/polarity/AMZN.US-with-polarity.csv', index=False)
    merged_data[merged_data["search_term"] == "TSLA"][modeling_columns].to_csv('data/polarity/TSLA.US-with-polarity.csv', index=False)
    merged_data[merged_data["search_term"] == "GOOG"][modeling_columns].to_csv('data/polarity/GOOG.US-with-polarity.csv', index=False)
    print("Data packaged...")


def build_results_graph():
    days=[1,2,3,4,5,6,7,8,9,10]
    accuracy_lg=[ 0.6943305,  0.6553399,  0.6866583,   0.6683352, 0.65596646, 0.652208999, 0.639210517, 0.6384286577, 0.63263272, 0.6326326]
    accuracy_svm=[0.6941726,  0.6302844,  0.6681800,   0.6747564, 0.69370480, 0.713434631, 0.752112553, 0.7397426234, 0.73128799, 0.7378627]
    accuracy_dt=[ 0.6384268,  0.5560588,  0.5602891,   0.5833037, 0.59129476, 0.606013213, 0.629349646, 0.6262127712, 0.66019414, 0.6576934]

    for index in range(10):
        accuracy_lg[index] = accuracy_lg[index] * 100
        accuracy_svm[index] = accuracy_svm[index] * 100
        accuracy_dt[index] = accuracy_dt[index] * 100

    plt.plot(days,accuracy_lg,'b-o',label='LG Accuracy over days', color="blue")
    plt.plot(days,accuracy_svm,'b-o',label='SVM Accuracy over days', color="red")
    plt.plot(days,accuracy_dt,'b-o',label='DT Accuracy over days', color="orange")
    plt.xlabel('Days')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    print('')


def evaluate_test_data(lg_estimator, svm_estimator, dt_estimator, X_test, y_test):
    print('Start evaluation on test data...')
    lg_y_pred = lg_estimator.predict(X_test)
    svm_y_pred = svm_estimator.predict(X_test)
    dt_y_pred = dt_estimator.predict(X_test)

    lg_f1_results = f1_score(y_test, lg_y_pred, average='weighted')
    svm_f1_results = f1_score(y_test, svm_y_pred, average='weighted')
    dt_f1_results = f1_score(y_test, dt_y_pred, average='weighted')

    lg_accuracy_results = accuracy_score(y_test, lg_y_pred, normalize=True)
    svm_accuracy_results = accuracy_score(y_test, svm_y_pred, normalize=True)
    dt_accuracy_results = accuracy_score(y_test, dt_y_pred, normalize=True)


    lg_recall_results = recall_score(y_test, lg_y_pred, average='weighted')
    svm_recall_results = recall_score(y_test, svm_y_pred, average='weighted')
    dt_recall_results = recall_score(y_test, dt_y_pred, average='weighted')

    lg_precision_results = precision_score(y_test, lg_y_pred, average='weighted')
    svm_precision_results = precision_score(y_test, svm_y_pred, average='weighted')
    dt_precision_results = precision_score(y_test, dt_y_pred, average='weighted')

    print(f'Logistic regression test data results: accuracy: {lg_accuracy_results}, f1: {lg_f1_results}, recall: {lg_recall_results}, precision: {lg_precision_results}')
    print(f'Support vector machine test data results: accuracy: {svm_accuracy_results}, f1: {svm_f1_results}, recall: {svm_recall_results}, precision: {svm_precision_results}')
    print(f'Decission tree test data results: accuracy: {dt_accuracy_results}, f1: {dt_f1_results}, recall: {dt_recall_results}, precision: {dt_precision_results}')

    print("Done evaluation on test data")


if __name__ == '__main__':
    print('Start')
    local = True
    merged_stocks = True
    #build_results_graph()
    if local:
        package_training_data(merged_stocks)
    else:
        download_data()
    stocks = read_data(merged_stocks)

    days_range = range(1, 11)
    for day in days_range:
        if merged_stocks:
            print("Start processing all stocks together ...")
            stock_name = "ALL"
            company_dfs = []
            for company in companies:
                processed_company_data = preprocess_data(stocks[stocks['search_term'] == company], day)
                company_dfs.append(processed_company_data)

            #keep_columns = ['date','close_value', 'stock_name', 'MACD', 'Signal_Line', 'RSI', 'SMA', 'EMA', 'agg_polarity']
            final_dataset = pd.concat(company_dfs)

            le_comp=LabelEncoder()
            final_dataset['stock_name'] = le_comp.fit_transform(final_dataset['search_term'])
            final_dataset['date'] = final_dataset['date'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))

            #X = final_dataset.drop(['Target', 'search_term', 'MACD', 'Signal_Line', 'RSI', 'SMA', 'EMA'], axis=1).values # features
            X = final_dataset.drop(['Target', 'search_term'], axis=1).values
            Y = final_dataset['Target'].values # labels
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

            lg_estimator, lg_accuracy = nested_cv_logistic_regression(X_train, y_train, stock_name)
            print(f'logistic regression for {stock_name} finished for target day after: {day}')
            svm_estimator, svm_accuracy = nested_cv_svm(X_train,y_train, stock_name)
            print(f'svm for {stock_name} finished for target day after: {day}')
            dt_estimator, dt_accuracy = nested_cv_decission_tree(X_train, y_train, stock_name)
            print(f'decission tree for {stock_name} finished for target day after: {day}')
            #build_results_graph()
            evaluate_test_data(lg_estimator, svm_estimator, dt_estimator, X_test,y_test )

        else:
            print("Start processing stock by stock ...")
            for stock in stocks:
                data = stock[0]
                stock_name = stock[1]
                X, Y = preprocess_data(data)
                lg_estimator, _ = nested_cv_logistic_regression(X, Y, stock_name)
                print(f'logistic regression for {stock_name} finished')
                svm_estimator, _ = nested_cv_svm(X,Y, stock_name)
                print(f'svm for {stock_name} finished')
                dt_estimator, _ = nested_cv_decission_tree(X, Y, stock_name)
                print(f'decission tree for {stock_name} finished')
                # save_models([
                #     {"lg":lg_estimator, "stock_name": stock_name},
                #     {"svm":svm_estimator, "stock_name": stock_name},
                #     {"dt":dt_estimator, "stock_name": stock_name}
                # ])