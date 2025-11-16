FROM python:3.10-slim-bookworm

WORKDIR /app

COPY . /app

# Install dependencies
RUN apt-get update -y && apt-get install -y awscli && \
    pip install --no-cache-dir -r requirement.txt

# Expose FastAPI port
EXPOSE 8080

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]