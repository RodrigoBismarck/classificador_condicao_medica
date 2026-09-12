FROM apache/airflow:2.9.0-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY airflow/requirements-airflow.txt /requirements-airflow.txt
RUN pip install --no-cache-dir -r /requirements-airflow.txt
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

ENV PYTHONPATH=/opt/airflow/project
WORKDIR /opt/airflow/project
