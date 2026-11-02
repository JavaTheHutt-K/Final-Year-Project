FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 user

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app/results \
    && chown -R user:user /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user backend ./backend
COPY --chown=user:user config.yaml .
COPY --chown=user:user data ./data

EXPOSE 7860

USER 1000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
