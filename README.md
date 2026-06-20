<<<<<<< HEAD
# Warehouse Route Optimizer

A production-capable warehouse order batching and route optimization platform with a React frontend, FastAPI backend, and research-focused experiment harness.

## Project Overview

This repository implements a warehouse order-picking optimizer for single-story warehouses. It combines:

- order batching based on spatial proximity, urgency, and capacity
- route planning using OR-Tools with aisle-aware graph traversal
- dynamic re-optimization for low-disruption updates
- a trainable travel-time estimator for realistic warehouse moves
- frontend visualization and control via React + Vite

The codebase supports both engineering-ready API usage and research experiments for comparing optimization strategies.

## Repository Structure

- `backend/`
  - `app/` — FastAPI service, optimization pipeline, schemas, and service orchestration
  - `requirements.txt` — Python dependencies for backend and experiments
  - `scripts/` — experiment and benchmark harness scripts
  - `tests/` — pytest coverage for optimizer behavior and API validation
- `frontend/`
  - React + TypeScript UI for simulation control and result display
  - Vite-based development and build tooling
- `artifacts/` — generated outputs and benchmark data (ignored in git)
- notebooks and research artifacts at the repository root

## Key Features

### Backend

- `backend/app/optimizer/graph_model.py`
  - constructs a warehouse grid graph and computes aisle-aware travel costs
- `backend/app/optimizer/batching.py`
  - capacity-aware order batching with spatial, urgency, and balance penalties
- `backend/app/optimizer/routing.py`
  - OR-Tools route solver for route sequencing
  - fallback routing behavior for robust solve execution
- `backend/app/optimizer/reoptimizer.py`
  - dynamic reoptimization with minimal task disruption
- `backend/app/ml/travel_time_model.py`
  - trainable travel-time estimator using scikit-learn
- `backend/app/services/optimizer_service.py`
  - orchestrates optimization, KPI evaluation, and baseline comparisons

### API

The backend exposes a versioned REST API under `/api/v1`.

- `GET /api/v1/health` — health check endpoint
- `POST /api/v1/optimize` — run warehouse optimization for a request payload
- `POST /api/v1/ml/train` — retrain the travel-time model from sample data
- `POST /api/v1/maps` — store a warehouse map layout
- `GET /api/v1/maps` — list stored warehouse maps
- `GET /api/v1/maps/{map_id}` — retrieve a saved map by ID

OpenAPI documentation is available at `http://127.0.0.1:8000/docs` when the backend is running.

### Frontend

- React + TypeScript application powered by Vite
- Supports local development and production build
- Provides controls for simulation execution and visualization of batch results

## Tech Stack

- Backend: Python, FastAPI, Pydantic, NetworkX, OR-Tools, scikit-learn
- Frontend: React, TypeScript, Vite
- Testing: pytest
- Containerization: Docker Compose

## Local Setup

### Backend

```powershell
cd backend
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

### Docker

```powershell
docker compose up --build
```

## Running Tests

```powershell
cd backend
pytest
```

## Experiments and Benchmarks

### Run the experiment script

```powershell
cd backend
python -m scripts.run_experiment
```

### Run benchmark harness

```powershell
cd backend
python -m scripts.benchmark --output-dir artifacts/benchmarks --scales 50 100 200 --seeds 1 2 3 4 5
```

Benchmark output includes:

- `artifacts/benchmarks/benchmark_runs.csv`
- `artifacts/benchmarks/benchmark_summary.json`

## Notes

- `artifacts/` is excluded from git by `.gitignore`.
- `.env` and local editor directories are ignored to keep the repository clean.
- The frontend is configured as a private Vite app in `frontend/package.json`.

## GitHub Repository

This project is hosted at:

https://github.com/Rakesh-experience-web/warehouse-route-optimizer
