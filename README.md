# Sim-Scene Review

Check a MuJoCo scene before you train or evaluate a robot policy in it.
Upload an MJCF file (or load the sample), watch three stages stream in,
then approve, reject or fix-and-rerun each object.

> **Status:** in development. The design is in
> [`docs/specs/`](docs/specs/2026-10-15-sim-scene-review-design.md) and the
> implementation plan in [`docs/plans/`](docs/plans/2026-10-15-sim-scene-review.md).
> The commands below work once the backend and frontend are implemented.

## Run it locally

Requirements: Python 3.9+ and Node 20+. The backend pins `mujoco==3.3.7`,
the newest release with a wheel for Python 3.9.

**1. Backend** (first terminal)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements-dev.txt   # macOS/Linux: .venv/bin/python
cd backend
../.venv/Scripts/python -m pytest -q                                  # optional: run the tests
../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

**2. Frontend** (second terminal)

```bash
cd frontend
npm install
npm test          # optional: run the tests
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`) and click
**Load sample**. Requests to `/api` are proxied to the backend on port 8000.

**Or with Docker** (one command, same image as the deployed service)

```bash
docker build -t sim-scene-review .
docker run --rm -p 8000:8000 sim-scene-review
```

Then open `http://localhost:8000`.
