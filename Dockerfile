FROM python:3.8-slim

# Memaksa Python mencetak log secara real-time tanpa penundaan buffer
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENTRYPOINT ["python", "-u", "train.py"]
