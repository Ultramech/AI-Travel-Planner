FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the healthchecks; the rest are wheels' build deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so code changes do not bust the layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY ui/ ./ui/
COPY scripts/ ./scripts/
# Streamlit reads its theme from .streamlit/config.toml relative to the working
# directory. Without this the containerised UI falls back to the stock light
# theme and the app's dark palette renders on a white page.
COPY .streamlit/ ./.streamlit/
# alembic/ and alembic.ini are required by the `migrate` compose service, which
# runs `alembic upgrade head` before the API and workers are allowed to start.
# Without them that service exits immediately and the whole stack stalls on a
# dependency that can never become healthy.
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY full_iata_codes.csv Station_Name_8477.xlsx ./
# The supplements that fill the gaps in those two datasets — Kerala's stations,
# Madgaon, Jaipur's airport. Omitting them does not fail the build: geo.py
# degrades quietly, so Docker would just answer "no railway station in the
# dataset for Kochi" while the same query worked locally. The `alembic/`
# directory was missed the same way once already.
COPY data/ ./data/

RUN mkdir -p /app/data/checkpoints \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
