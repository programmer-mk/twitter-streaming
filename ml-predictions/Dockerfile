FROM python:3.8.2
# try this: smizy/scikit-learn

RUN mkdir ml-predictions
ENV MODEL_DIR=/home/jovyan/ml-predictions

ENV LG_MODEL_FILE_MSFT=../lg-MSFT-estimator.pkl
ENV SVM_MODEL_FILE_MSFT=../svm-MSFT-estimator.pkl
ENV DT_MODEL_FILE_MSFT=../dt-MSFT-estimator.pkl

ENV LG_MODEL_FILE_AMZN=../lg-AMZN-estimator.pkl
ENV SVM_MODEL_FILE_AMZN=../svm-AMZN-estimator.pkl
ENV DT_MODEL_FILE_AMZN=../dt-AMZN-estimator.pkl

ENV LG_MODEL_FILE_AAPL=../lg-AAPL-estimator.pkl
ENV SVM_MODEL_FILE_AAPL=../svm-AAPL-estimator.pkl
ENV DT_MODEL_FILE_AAPL=../dt-AAPL-estimator.pkl

ENV LG_MODEL_FILE_TSLA=../lg-TSLA-estimator.pkl
ENV SVM_MODEL_FILE_TSLA=../svm-TSLA-estimator.pkl
ENV DT_MODEL_FILE_TSLA=../dt-TSLA-estimator.pkl

ENV LG_MODEL_FILE_BTC=../lg-BTC-estimator.pkl
ENV SVM_MODEL_FILE_BTC=../svm-BTC-estimator.pkl
ENV DT_MODEL_FILE_BTC=../dt-BTC-estimator.pkl

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# next line is just for testing purposies, remove at some point
COPY ./data/test.csv /home/jovyan/ml-predictions/data/test.csv

COPY start-up.sh ./start-up.sh
COPY training.py ./training.py
COPY api.py ./api.py

#RUN chmod +w /home/jovyan/ml-predictions/data/test.csv
#RUN chmod +x api.py
#RUN python3 training.py
#RUN chmod 777 ./start-up.sh
RUN chmod +x start-up.sh
CMD sh ./start-up.sh