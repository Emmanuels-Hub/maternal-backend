FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY manage.py n_atlas_model.py ./

EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn manage:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 180 --access-logfile -"]