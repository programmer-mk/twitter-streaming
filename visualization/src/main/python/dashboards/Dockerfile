FROM python:3.9.1
ADD . /visualization
WORKDIR /visualization
RUN pip install -r requirements.txt
RUN chmod +x app.py
CMD python app.py