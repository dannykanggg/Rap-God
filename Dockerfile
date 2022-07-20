FROM python:3.9.7
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
#CMD streamlit run app.py
CMD streamlit run --server.port $PORT app.py