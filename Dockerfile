FROM python:3.8

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
