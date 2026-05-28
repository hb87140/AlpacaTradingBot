FROM python:3.13-slim

WORKDIR /app

# Install gcc for any packages that compile C extensions
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the log directory exists at build time
RUN mkdir -p logs

EXPOSE 8080

CMD ["python", "main.py"]
