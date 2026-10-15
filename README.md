# Sim-Scene Review

Check a MuJoCo scene before you train or evaluate a robot policy in it.
Upload an MJCF file (or load the sample), watch three stages stream in,
then approve, reject or fix-and-rerun each object.

**Live:** _(URL added after the first deploy)_

## What it checks

| Stage | What happens | Findings |
| -- | -- | -- |
| Load & compile | MuJoCo compiles the scene. Compile errors end the run with the compiler's message. | — |
| Static asset checks | Rules over the compiled model, one object at a time. | `zero_mass`, `no_collision`, `default_friction`, `unlimited_joint`, `implausible_scale` |
| Simulation check | 2 s of simulated time, capped at 10 s wall. | `initial_penetration`, `exploded`, `fell_through_floor`, `unstable` |

Four findings carry a one-click fix (recompute mass, enable collision, add
joint limits, scale millimetres to metres). A fix edits the scene through
MuJoCo's `MjSpec`, the result is compiled fresh, and only that object is
re-checked. There is no inertia rule: MuJoCo's compiler already refuses bad
inertia and zero mass, so those surface at stage 1.

## How it is built

- **Backend** — FastAPI. The scene is compiled once with `mujoco`; the checks
  and the geometry the browser draws read that one compiled model, so what
  you see is what is simulated. Fixes edit the XML through `MjSpec` and the
  result is compiled fresh. Rules are pure functions;
  one that throws becomes a `check_error` finding instead of killing the run.
- **Streaming** — each run keeps its events in memory with increasing ids.
  The browser's `EventSource` reconnects on its own and sends
  `Last-Event-ID`; the server replays from there. Waiting is a blocking
  `Condition.wait_for` in a worker thread, so there is no polling loop.
- **Idempotency** — a fix request is keyed by `scene version + object + fix`,
  recorded for both the version the fix was applied to and the version it
  produced. A rerun finishes in about 100 ms, so a second press can't race
  the first; instead the object panel lists every fix applied in this session
  with a Re-apply button. Pressing it sends the same fix on the same version,
  the server returns the original run, charges nothing, and the footer's
  "saved" figure moves.
- **Frontend** — Vite, React, TypeScript, React Three Fiber. Events fold
  into view state through a pure reducer (`src/reducer.ts`, unit-tested);
  after each run the page re-reads the server snapshot so decisions and the
  scene version are authoritative.
- **One image** — Node builds the frontend; `python:3.12-slim` serves the
  API, `index.html` and `/assets` from one origin.

## What is measured

- **Time to usable scene** — from scene creation until every object is
  approved. This is the number I would watch first in a real product.
- **Manual fixes** and **reruns** — how much human and compute effort a
  scene needed.
- **Cost** — from a fake rate card (compile $0.001/run, static $0.0005/object,
  simulation $0.01/simulated second) so the idempotency saving is visible.

## Deliberate trade-offs

- State is in memory: 200 scenes, two-hour TTL, gone on restart. A real
  service would persist runs and let the SSE buffer be rebuilt from storage.
- Uploads must be a single MJCF file under 1 MB with no `<include>` and no
  `file=` attributes, so the server never opens a path an upload names.
  Inline mesh vertices still work.
- An object rerun re-checks only that object. Other objects keep their
  earlier findings even though the fix changed the scene; **Run again**
  does the full pass.
- The mm→m fix scales only the named body's geoms (sizes and positions).
  Child bodies, joint positions and the body's own position are untouched, so
  a multi-body object modelled in millimetres needs a full fix by hand.
- The "Applied fixes" list is page-session state; a reload clears it (the
  server still remembers the idempotency keys).
- The simulation rules use thresholds (20 m/s, 0.5 m below the floor) rather
  than physics-aware judgement. They catch the gross failures a person would
  otherwise find by eye in the viewer.
- Canonicalisation re-serialises the upload through MuJoCo, which rounds
  every number to 6 significant digits.
- No rate limiting. A scene keeps only its last 20 runs.
- Only MJCF. USD and zip uploads with mesh files are the obvious next inputs.
- No end-to-end browser tests; the backend (76 tests) and the frontend (16
  tests, including the reducer) are unit-tested.

## Sample scene

`backend/app/samples/tabletop.xml` plants: a massless mug, a plate with
collision off, a box starting inside the table top (with default friction),
a cabinet door hinge without limits, and a tray modelled in millimetres. The
tray sits 200 m away so its 1000× size does not swallow the table until you
fix it. `mug_b` is the control.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements-dev.txt   # macOS/Linux: .venv/bin/python
cd backend
../.venv/Scripts/python -m pytest -q
../.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
# second shell
cd frontend && npm install && npm test && npm run dev
```

Open the Vite URL; `/api` is proxied to port 8000.

Or run the deployed image locally:

```bash
docker build -t sim-scene-review .
docker run --rm -p 8000:8000 sim-scene-review
```

Then open `http://localhost:8000`.

## Deploy

`render.yaml` describes one free Docker web service on Render. Cold starts
on the free plan take about 30 s. Health check: `GET /api/health`.

## Note on tooling

Built with Claude Code as a pair; every design decision, test and trade-off
above was reviewed and is mine to defend.
