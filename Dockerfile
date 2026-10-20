FROM python:3.8-slim

# Sesuai dokumentasi: Dilarang memakai /home atau /data
WORKDIR /app

# Install dependency saat build image (container terisolasi dari internet saat runtime)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

# Wajib menggunakan ENTRYPOINT dengan flag -u
ENTRYPOINT ["python", "-u", "train.py"]

