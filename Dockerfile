FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Menghapus karakter CRLF (\r) Windows secara otomatis di Linux container
RUN sed -i 's/\r$//' train.py

ENTRYPOINT ["python", "-u", "train.py"]
