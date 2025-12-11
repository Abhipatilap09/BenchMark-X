FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# ensure static subdirectories are directories (repo contains zero-byte placeholder files)
RUN rm -f static/uploads static/plots static/architectures || true && \
    mkdir -p static/uploads static/plots static/architectures

EXPOSE 5000

CMD ["python", "app.py"]
