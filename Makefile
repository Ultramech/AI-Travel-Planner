.PHONY: help install api worker ui test lint infra up down logs clean graph \
        migrate migration db-shell db-reset test-db check-db

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dependencies
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "Done. Fill in .env before running."

api: ## Run the FastAPI service (http://localhost:8000/docs)
	.venv/bin/uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Run a Kafka worker
	$(PY) -m app.workers.consumer

ui: ## Run the Streamlit client (http://localhost:8501)
	.venv/bin/streamlit run ui/streamlit_app.py

test: ## Run the test suite (never touches a real database)
	$(PY) -m pytest tests/ -v

test-db: ## Run the suite including Postgres integration tests
	# Note the _test database, on the local scratch Postgres — NEVER Supabase.
	# Point this at your application database and the suite will write fixture
	# trips into it. That container is profiled off, so start it explicitly.
	docker compose --profile testdb up -d postgres
	TEST_DATABASE_URL="$${TEST_DATABASE_URL:-postgresql://planner:planner@localhost:5434/travel_planner_test}" \
	  $(PY) -m pytest tests/ -v

infra: ## Start only Redis + Kafka + Selenium (the database is Supabase)
	docker compose up -d redis kafka selenium

check-db: ## Diagnose DATABASE_URL (connection, TLS, schema)
	PYTHONPATH=. $(PY) scripts/check_db.py

migrate: ## Apply database migrations (needs DATABASE_URL)
	PYTHONPATH=. .venv/bin/alembic upgrade head

migration: ## Autogenerate a migration:  make migration m="add x"
	PYTHONPATH=. .venv/bin/alembic revision --autogenerate -m "$(m)"

db-shell: ## Open psql against DATABASE_URL
	@psql "$$(grep -E '^DATABASE_URL=' .env | cut -d= -f2-)"

db-reset: ## Drop and recreate every table (DESTRUCTIVE)
	@echo "This deletes all trip history. Ctrl-C within 5s to abort."; sleep 5
	PYTHONPATH=. .venv/bin/alembic downgrade base && PYTHONPATH=. .venv/bin/alembic upgrade head

up: ## Start the whole stack
	docker compose up --build

down: ## Stop everything
	docker compose down

logs: ## Tail worker logs
	docker compose logs -f worker

graph: ## Print the graph topology as mermaid
	$(PY) -c "import asyncio;from app.graph.builder import export_mermaid;print(asyncio.run(export_mermaid()))"

clean: ## Remove caches and local checkpoints
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache data/checkpoints
