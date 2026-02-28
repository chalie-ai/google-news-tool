FROM python:3.11-slim
WORKDIR /tool
COPY requirements.txt ./
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
ENTRYPOINT ["python", "runner.py"]
