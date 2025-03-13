FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y cmake libboost-all-dev \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
