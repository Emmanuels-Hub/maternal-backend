FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

COPY requirements-space.txt ./requirements-space.txt
RUN pip install --no-cache-dir -r requirements-space.txt

COPY app ./app
COPY manage.py n_atlas_model.py ./

EXPOSE 7860

CMD ["gunicorn", "manage:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--access-logfile", "-"]