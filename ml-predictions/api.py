# We now need the json library so we can load and export json data
import json
import os
import pandas as pd
from joblib import load
from sklearn import preprocessing

from flask import Flask

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
LG_MODEL_FILE_MSFT = os.environ["LG_MODEL_FILE_MSFT"]
SVM_MODEL_FILE_MSFT = os.environ["SVM_MODEL_FILE_MSFT"]
DT_MODEL_FILE_MSFT = os.environ["DT_MODEL_FILE_MSFT"]

MSFT_LG_MODEL_PATH = os.path.join(MODEL_DIR, LG_MODEL_FILE_MSFT)
MSFT_SVM_MODEL_PATH = os.path.join(MODEL_DIR, SVM_MODEL_FILE_MSFT)
MSFT_DT_MODEL_PATH = os.path.join(MODEL_DIR, DT_MODEL_FILE_MSFT)

# Loading logistic regression model for microsoft data
print("Loading microsoft logistic regression model from: {}".format(MSFT_LG_MODEL_PATH))
msft_lg_model = load(MSFT_LG_MODEL_PATH)

# Loading support vector machine model for microsoft data
print("Loading microsoft support vector machine model from: {}".format(MSFT_SVM_MODEL_PATH))
msft_svm_model = load(MSFT_SVM_MODEL_PATH)

# Loading decision tree  model for microsoft data
print("Loading microsoft decision tree model from: {}".format(MSFT_DT_MODEL_PATH))
msft_dt_model = load(MSFT_DT_MODEL_PATH)

# Creation of the Flask app
app = Flask(__name__)


# Flask route so that we can serve HTTP traffic on that route
@app.route('/line/<Line>')
# Get data from json and return the requested row defined by the variable Line
def line(Line):
    with open('./test.json', 'r') as jsonfile:
        file_data = json.loads(jsonfile.read())
    return json.dumps(file_data[Line])


# Flask route so that we can serve HTTP traffic on that route
@app.route('/prediction/<int:Line>', methods=['POST', 'GET'])
def prediction(Line):
    data = pd.read_json('./test.json')
    data_test = data.transpose()
    #X = data_test.drop(data_test.loc[:, 'Line':'# Letter'].columns, axis = 1)
    X_test = data_test.iloc[Line,:].values.reshape(1, -1)

    #msft_lg_model = load(MSFT_LG_MODEL_PATH) --> should not be needed
    prediction_msft_lg = msft_lg_model.predict(X_test)

    #msft_svm_model = load(MSFT_SVM_MODEL_PATH) --> should not be needed
    prediction_msft_svm = msft_svm_model.predict(X_test)

    #msft_dt_model = load(MSFT_DT_MODEL_PATH) --> should not be needed
    prediction_msft_dt = msft_dt_model.predict(X_test)

    return {
        'prediction logistic regression': int(prediction_msft_lg),
        'prediction support vector machine': int(prediction_msft_svm),
        'prediction decission tree': int(prediction_msft_dt)
    }


@app.route('/score',methods=['POST', 'GET'])
# Return classification score for both Neural Network and LDA inference model from the all dataset provided
def score():
    data_test = pd.read_csv('./test.csv')
    y_test = data_test['Target'].values
    X_test = data_test.drop(data_test.loc[:, 'Target'].columns, axis = 1)

    msft_lg_score = msft_lg_model.score(X_test, y_test)
    msft_svm_score = msft_svm_model.score(X_test, y_test)
    msft_dt_score = msft_dt_model.score(X_test, y_test)

    return {
        'score logistic regression': msft_lg_score,
        'scpre support vector machine': msft_svm_score,
        'score decission tree': msft_dt_score
    }


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')