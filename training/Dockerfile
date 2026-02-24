FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY training_pipeline.py .
COPY utils.py .
CMD ["python", "training_pipeline.py"]
