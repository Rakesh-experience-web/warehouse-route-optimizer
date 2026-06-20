# Warehouse Batch Route Optimizer

Production-oriented, algorithm-heavy platform for single-story warehouse order clustering and batch route optimization.

## Implemented Architecture

- `backend/app/optimizer/graph_model.py`
  - Grid warehouse graph generation.
  - Shortest path support for aisle travel costs.
- `backend/app/optimizer/batching.py`
  - Capacity-constrained K-medoids style order batching using:
    - spatial proximity
    - due-time urgency penalty
    - load balancing penalty
- `backend/app/optimizer/routing.py`
  - OR-Tools route solver (TSP-style) with nearest-neighbor fallback.
  - Path expansion from route nodes into movement and pick steps.
- `backend/app/optimizer/reoptimizer.py`
  - Low-disruption dynamic re-optimization policy.
- `backend/app/ml/travel_time_model.py`
  - Trainable travel-time estimator (RandomForest baseline).
- `backend/app/services/optimizer_service.py`
  - End-to-end orchestration and KPI evaluation against naive baseline.
- `frontend/src/App.tsx`
  - Control panel for simulation runs and KPI/batch results visualization.

## Tech Stack

- Frontend: React + TypeScript + Vite
- Backend: FastAPI + Pydantic
- Optimization: NetworkX + OR-Tools + custom heuristics
- ML: scikit-learn
- Testing: pytest

## Local Run

### 1) Backend

```powershell
cd backend
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 2) Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Docker Run

```powershell
docker compose up --build
```

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/optimize`
- `POST /api/v1/ml/train`

OpenAPI:

- `http://127.0.0.1:8000/docs`

## Test

```powershell
cd backend
pytest
```

## Research Experiment Script

```powershell
cd backend
python -m scripts.run_experiment
```

## Research Benchmark Harness (Baselines + Ablations)

Runs multi-scenario, multi-seed experiments and writes CSV/JSON artifacts for paper figures.

```powershell
cd backend
python -m scripts.benchmark --output-dir artifacts/benchmarks --scales 50 100 200 --seeds 1 2 3 4 5
```

Outputs:
- `artifacts/benchmarks/benchmark_runs.csv`
- `artifacts/benchmarks/benchmark_summary.json`
- `artifacts/benchmarks/ablation_runs.csv`
- `artifacts/benchmarks/ablation_summary.json`

Strategies included:
- `full` (current production optimizer)
- `greedy_nn` (greedy capacity batching + nearest-neighbor routing)
- `spatial_ortools` (spatial-only batching objective + OR-Tools routing)
