FROM jupyter/scipy-notebook

RUN mkdir ml-predictions
ENV MODEL_DIR=/home/jovyan/ml-predictions
ENV LG_MODEL_FILE_MSFT=../lg-estimator.pkl
ENV SVM_MODEL_FILE_MSFT=../svm-estimator.pkl
ENV DT_MODEL_FILE_MSFT=../dt-estimator.pkl

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./data/MSFT.US-historical-stocks.csv /home/jovyan/ml-predictions/data/MSFT.US-historical-stocks.csv
COPY ./data/test.csv /home/jovyan/ml-predictions/data/test.csv

COPY training.py ./training.py
COPY api.py ./api.py

#RUN chmod +w /home/jovyan/ml-predictions/data/test.csv

RUN python3 training.py