FROM python:3.9.7
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /app
ENTRYPOINT [ "streamlit", "run" ]
CMD ["sh", "-c","streamlit run --server.port $PORT app.py"]