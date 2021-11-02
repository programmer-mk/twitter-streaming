import os
import pandas as pd
from joblib import load
from sklearn import preprocessing
import json
from flask import Flask,request, jsonify

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
LG_MODEL_FILE_MSFT = os.environ["LG_MODEL_FILE_MSFT"]
SVM_MODEL_FILE_MSFT = os.environ["SVM_MODEL_FILE_MSFT"]
DT_MODEL_FILE_MSFT = os.environ["DT_MODEL_FILE_MSFT"]

LG_MODEL_FILE_AMZN = os.environ["LG_MODEL_FILE_AMZN"]
SVM_MODEL_FILE_AMZN = os.environ["SVM_MODEL_FILE_AMZN"]
DT_MODEL_FILE_AMZN = os.environ["DT_MODEL_FILE_AMZN"]

LG_MODEL_FILE_AAPL = os.environ["LG_MODEL_FILE_AAPL"]
SVM_MODEL_FILE_AAPL = os.environ["SVM_MODEL_FILE_AAPL"]
DT_MODEL_FILE_AAPL = os.environ["DT_MODEL_FILE_AAPL"]

LG_MODEL_FILE_TSLA = os.environ["LG_MODEL_FILE_TSLA"]
SVM_MODEL_FILE_TSLA = os.environ["SVM_MODEL_FILE_TSLA"]
DT_MODEL_FILE_TSLA = os.environ["DT_MODEL_FILE_TSLA"]

LG_MODEL_FILE_BTC = os.environ["LG_MODEL_FILE_BTC"]
SVM_MODEL_FILE_BTC = os.environ["SVM_MODEL_FILE_BTC"]
DT_MODEL_FILE_BTC = os.environ["DT_MODEL_FILE_BTC"]

MSFT_LG_MODEL_PATH = os.path.join(MODEL_DIR, LG_MODEL_FILE_MSFT)
MSFT_SVM_MODEL_PATH = os.path.join(MODEL_DIR, SVM_MODEL_FILE_MSFT)
MSFT_DT_MODEL_PATH = os.path.join(MODEL_DIR, DT_MODEL_FILE_MSFT)

AMZN_LG_MODEL_PATH = os.path.join(MODEL_DIR, LG_MODEL_FILE_AMZN)
AMZN_SVM_MODEL_PATH = os.path.join(MODEL_DIR, SVM_MODEL_FILE_AMZN)
AMZN_DT_MODEL_PATH = os.path.join(MODEL_DIR, DT_MODEL_FILE_AMZN)

AAPL_LG_MODEL_PATH = os.path.join(MODEL_DIR, LG_MODEL_FILE_AAPL)
AAPL_SVM_MODEL_PATH = os.path.join(MODEL_DIR, SVM_MODEL_FILE_AAPL)
AAPL_DT_MODEL_PATH = os.path.join(MODEL_DIR, DT_MODEL_FILE_AAPL)

TSLA_LG_MODEL_PATH = os.path.join(MODEL_DIR, LG_MODEL_FILE_TSLA)
TSLA_SVM_MODEL_PATH = os.path.join(MODEL_DIR, SVM_MODEL_FILE_TSLA)
TSLA_DT_MODEL_PATH = os.path.join(MODEL_DIR, DT_MODEL_FILE_TSLA)

BTC_LG_MODEL_PATH = os.path.join(MODEL_DIR, LG_MODEL_FILE_BTC)
BTC_SVM_MODEL_PATH = os.path.join(MODEL_DIR, SVM_MODEL_FILE_BTC)
BTC_DT_MODEL_PATH = os.path.join(MODEL_DIR, DT_MODEL_FILE_BTC)

print("Loading Microsoft logistic regression model from: {}".format(MSFT_LG_MODEL_PATH))
msft_lg_model = load(MSFT_LG_MODEL_PATH)
print("Loading Microsoft support vector machine model from: {}".format(MSFT_SVM_MODEL_PATH))
msft_svm_model = load(MSFT_SVM_MODEL_PATH)
print("Loading Microsoft decision tree model from: {}".format(MSFT_DT_MODEL_PATH))
msft_dt_model = load(MSFT_DT_MODEL_PATH)

print("Loading Amazon logistic regression model from: {}".format(AMZN_LG_MODEL_PATH))
amzn_lg_model = load(AMZN_LG_MODEL_PATH)
print("Loading Amazon support vector machine model from: {}".format(AMZN_SVM_MODEL_PATH))
amzn_svm_model = load(AMZN_SVM_MODEL_PATH)
print("Loading Amazon decision tree model from: {}".format(AMZN_DT_MODEL_PATH))
amzn_dt_model = load(AMZN_DT_MODEL_PATH)

print("Loading Apple logistic regression model from: {}".format(AAPL_LG_MODEL_PATH))
aapl_lg_model = load(AAPL_LG_MODEL_PATH)
print("Loading Apple support vector machine model from: {}".format(AAPL_SVM_MODEL_PATH))
aapl_svm_model = load(AAPL_SVM_MODEL_PATH)
print("Loading Apple decision tree model from: {}".format(AAPL_DT_MODEL_PATH))
aapl_dt_model = load(AAPL_DT_MODEL_PATH)

print("Loading Tesla logistic regression model from: {}".format(TSLA_LG_MODEL_PATH))
tsla_lg_model = load(TSLA_LG_MODEL_PATH)
print("Loading Tesla support vector machine model from: {}".format(TSLA_SVM_MODEL_PATH))
tsla_svm_model = load(TSLA_SVM_MODEL_PATH)
print("Loading Tesla decision tree model from: {}".format(TSLA_DT_MODEL_PATH))
tsla_dt_model = load(TSLA_DT_MODEL_PATH)

print("Loading Bitcoin logistic regression model from: {}".format(BTC_LG_MODEL_PATH))
btc_lg_model = load(BTC_LG_MODEL_PATH)
print("Loading Bitcoin support vector machine model from: {}".format(BTC_SVM_MODEL_PATH))
btc_svm_model = load(BTC_SVM_MODEL_PATH)
print("Loading Bitcoin decision tree model from: {}".format(BTC_DT_MODEL_PATH))
btc_dt_model = load(BTC_DT_MODEL_PATH)

# Creation of the Flask app
app = Flask(__name__)

@app.route('/line/<Line>')
def line(Line):
    data = pd.read_csv('/home/jovyan/ml-predictions/data/test.csv')
    target_line = data[data['Index'] == int(Line)]
    return {
        "data": json.loads(target_line.to_json(orient='records'))[0]
    }


# Flask route so that we can serve HTTP traffic on that route
@app.route('/prediction/<int:Line>', methods=['POST', 'GET'])
def prediction(Line):
    Line = int(Line)
    data = pd.read_csv('/home/jovyan/ml-predictions/data/test.csv')
    target_line = data[data['Index'] == Line].drop(['Index'], axis=1)
    print(f'target line: {target_line}')
    Y_test = target_line['Target'].values
    X_test = target_line.drop(['Target'], axis=1).values

    print(f'Real value: {Y_test}')

    prediction_msft_lg = msft_lg_model.predict(X_test)
    prediction_msft_svm = msft_svm_model.predict(X_test)
    prediction_msft_dt = msft_dt_model.predict(X_test)

    prediction_amzn_lg = amzn_lg_model.predict(X_test)
    prediction_amzn_svm = amzn_svm_model.predict(X_test)
    prediction_amzn_dt = amzn_dt_model.predict(X_test)

    prediction_aapl_lg = aapl_lg_model.predict(X_test)
    prediction_aapl_svm = aapl_svm_model.predict(X_test)
    prediction_aapl_dt = aapl_dt_model.predict(X_test)

    prediction_tsla_lg = tsla_lg_model.predict(X_test)
    prediction_tsla_svm = tsla_svm_model.predict(X_test)
    prediction_tsla_dt = tsla_dt_model.predict(X_test)

    prediction_btc_lg = btc_lg_model.predict(X_test)
    prediction_btc_svm = btc_svm_model.predict(X_test)
    prediction_btc_dt = btc_dt_model.predict(X_test)

    return {
        'stock_predictions': [{
            'microsoft': {
                'logistic regression': int(prediction_msft_lg),
                'support vector machine': int(prediction_msft_svm),
                'decission tree': int(prediction_msft_dt)
            },
            'amazon': {
                'logistic regression': int(prediction_amzn_lg),
                'support vector machine': int(prediction_amzn_svm),
                'decission tree': int(prediction_amzn_dt)
            },
            'apple': {
                'logistic regression': int(prediction_aapl_lg),
                'support vector machine': int(prediction_aapl_svm),
                'decission tree': int(prediction_aapl_dt)
            },
            'tesla': {
                'logistic regression': int(prediction_tsla_lg),
                'support vector machine': int(prediction_tsla_svm),
                'decission tree': int(prediction_tsla_dt)
            },
            'bitcoin': {
                'logistic regression': int(prediction_btc_lg),
                'support vector machine': int(prediction_btc_svm),
                'decission tree': int(prediction_btc_dt)
            }
        }]
    }


@app.route('/score',methods=['POST', 'GET'])
def score():
    data = pd.read_csv('/home/jovyan/ml-predictions/data/test.csv')
    y_test = data['Target'].values
    X_test = data.drop(['Target', 'Index'], axis=1)

    msft_lg_score = msft_lg_model.score(X_test, y_test)
    msft_svm_score = msft_svm_model.score(X_test, y_test)
    msft_dt_score = msft_dt_model.score(X_test, y_test)

    prediction_amzn_lg = amzn_lg_model.predict(X_test)
    prediction_amzn_svm = amzn_svm_model.predict(X_test)
    prediction_amzn_dt = amzn_dt_model.predict(X_test)

    prediction_aapl_lg = aapl_lg_model.predict(X_test)
    prediction_aapl_svm = aapl_svm_model.predict(X_test)
    prediction_aapl_dt = aapl_dt_model.predict(X_test)

    prediction_tsla_lg = tsla_lg_model.predict(X_test)
    prediction_tsla_svm = tsla_svm_model.predict(X_test)
    prediction_tsla_dt = tsla_dt_model.predict(X_test)

    prediction_btc_lg = btc_lg_model.predict(X_test)
    prediction_btc_svm = btc_svm_model.predict(X_test)
    prediction_btc_dt = btc_dt_model.predict(X_test)

    return {
        'stock_predictions': [{
            'microsoft': {
                'logistic regression': int(prediction_msft_lg),
                'support vector machine': int(prediction_msft_svm),
                'decission tree': int(prediction_msft_dt)
            },
            'amazon': {
                'logistic regression': int(prediction_amzn_lg),
                'support vector machine': int(prediction_amzn_svm),
                'decission tree': int(prediction_amzn_dt)
            },
            'apple': {
                'logistic regression': int(prediction_aapl_lg),
                'support vector machine': int(prediction_aapl_svm),
                'decission tree': int(prediction_aapl_dt)
            },
            'tesla': {
                'logistic regression': int(prediction_tsla_lg),
                'support vector machine': int(prediction_tsla_svm),
                'decission tree': int(prediction_tsla_dt)
            },
            'bitcoin': {
                'logistic regression': int(prediction_btc_lg),
                'support vector machine': int(prediction_btc_svm),
                'decission tree': int(prediction_btc_dt)
            }
        }]
    }

@app.route('/tweets',methods=['POST'])
def tweets():
    content = request.json
    print(content)
    return {
        "status": "ok"
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)