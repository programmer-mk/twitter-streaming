import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline


def load_data():
    return pd.read_csv('data/MSFT.US-historical-stocks.csv')


def SMA(data, period=30, column='Close'):
    return data[column].rolling(window=period).mean()


def EMA(data, period=20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()


def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    short_ema = EMA(data, period_short, column=column)
    long_ema = EMA(data, period_long, column=column)
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = EMA(data, period_signal, column='MACD')
    return data


def RSI(data, period = 14, column = 'Close'):
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
            joblib.dump(estimator[estimator_key], f'{estimator_key}-estimator.pkl', compress=1)


def preprocess_data(data):
    MACD(data)
    RSI(data)
    data['SMA'] = SMA(data)
    data['EMA'] = EMA(data)
    data = data.dropna()

    #Create the target column
    data['Target'] = np.where(data['Close'].shift(-7) > data['Close'], 1, 0) # if before 7 days price is lower than todays price put 1 else put 0

    data_build = data[(data['Date'] >= '2016-01-01') & (data['Date'] < '2021-07-24')]
    data_verification = data[(data['Date'] >= '2021-07-24') & (data['Date'] < '2021-08-29')]

    keep_columns = ['Close', 'MACD', 'Signal_Line', 'RSI', 'SMA', 'EMA']
    keep_columns_test = ['Close', 'MACD', 'Signal_Line', 'RSI', 'SMA', 'EMA', 'Target']
    data_verification[keep_columns_test].to_csv('data/test.csv', index=True, index_label='Index')
    features = data_build[keep_columns].values
    labels = data_build['Target'].values
    return features, labels


def nested_cv_logistic_regression(X, Y):
    param_grid = {"logisticregression__C": [0.0001, 0.0005, 0.001, 0.01, 0.3, 0.5, 1, 5, 10],
              "logisticregression__penalty":["l1","l2"],
              "logisticregression__solver": ["liblinear", "saga"]
              }

    lgr = LogisticRegression(max_iter=20000)
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('logisticregression', lgr)])

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Nested CV with parameter optimization
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv)
    nested_scores = cross_val_score(clf, X=X, y=Y, scoring='accuracy', cv=outer_cv)
    nested_score = nested_scores.mean()

    print(f'Results after cross-validation logistic regression {nested_score} \n')

    clf.fit(X, Y)
    print(f'best estimator: {clf.best_estimator_}')
    print(f'best params: {clf.best_params_}')
    return clf


def nested_cv_svm(X, Y):
    param_grid = {'svm__C': [0.0001, 0.0005, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
                  'svm__gamma': [100, 10, 1, 0.1, 0.01, 0.001],
                  'svm__kernel': ['rbf']}

    svm = SVC()
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('svm', svm)])

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Nested CV with parameter optimization
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
    nested_scores = cross_val_score(clf, X=X, y=Y, scoring='accuracy', cv=outer_cv)
    nested_score = nested_scores.mean()

    print(f'Results after cross-validation support vector machine {nested_score} \n')

    clf.fit(X, Y)
    print(f'best estimator: {clf.best_estimator_}')
    print(f'best params: {clf.best_params_}')

    return clf


def nested_cv_decission_tree(X, Y):
    param_grid = {'tree__max_features': ['auto', 'sqrt', 'log2'],
                  'tree__ccp_alpha': [0.1, .01, .001],
                  'tree__max_depth' : [5, 6, 7, 8, 9, 12, 15, 18],
                  'tree__criterion' :['gini', 'entropy']
                  }

    tree_clas = DecisionTreeClassifier(random_state=1024)
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('tree', tree_clas)])
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

    # Nested CV with parameter optimization
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=inner_cv, n_jobs=-1)
    nested_scores = cross_val_score(clf, X=X, y=Y, scoring='accuracy', cv=outer_cv)
    nested_score = nested_scores.mean()

    print(f'Results after cross-validation decission tree {nested_score} \n')

    clf.fit(X, Y)
    print(f'best estimator: {clf.best_estimator_}')
    print(f'best params: {clf.best_params_}')

    return clf


if __name__ == '__main__':
    data = load_data()
    X, Y = preprocess_data(data)
    lg_estimator = nested_cv_logistic_regression(X, Y)
    svm_estimator = nested_cv_svm(X,Y)
    dt_estimator = nested_cv_decission_tree(X, Y)
    save_models([
        {"lg":lg_estimator},
        {"svm":svm_estimator},
        {"dt":dt_estimator}
    ])