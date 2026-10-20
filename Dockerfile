FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install utility penanganan karakter baris Windows (CRLF)
RUN apt-get update && apt-get install -y --no-install-recommends dos2unix && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN dos2unix train.py

ENTRYPOINT ["python", "-u", "train.py"]
