FROM python:3.10-slim-bookworm

WORKDIR /app

COPY . /app

# Install AWS CLI
RUN apt-get update -y && apt-get install -y awscli

RUN pip install --no-cache-dir -r requirement.txt

CMD ["python", "app.py"]
