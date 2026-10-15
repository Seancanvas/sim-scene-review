# Sim-Scene Review — Design

Date: 2026-10-15
Status: approved design, pending spec review

## Purpose

A small, public, end-to-end product that shows how a person reviews a
simulation scene before a robot policy is trained or evaluated in it. It is a
portfolio piece: a reviewer should be able to open one URL, watch a sample
scene go through three checking stages in under 30 seconds, act on the
findings, and understand the architecture and trade-offs from the README in
three minutes.

It does **not** reconstruct scenes from video or images. It checks MJCF scenes
that already exist and tells the user what would make them unsafe to simulate.

## Success criteria

1. Opening the deployed URL and choosing the sample scene streams all three
   stages and lists every planted issue (see § Sample scene).
2. Every object can be approved, rejected, or fixed-and-rerun on its own.
3. Pressing the same fix twice does not run or charge twice; the UI says so.
4. Reloading the page mid-run restores the state; dropping the stream and
   reconnecting loses no events.
5. The README explains the architecture, the measurements, and the deliberate
   omissions.

## Scope

In:

- Input: the built-in sample scene, or one uploaded MJCF XML file (≤ 1 MB).
- Three stages per run:
  1. **Load & compile** — compile with MuJoCo `MjSpec`; compile errors become a
     displayable failure.
  2. **Static asset checks** — per object rules over the compiled model.
  3. **Simulation check** — step 2 s of simulated time and inspect what moved.
- Per-object actions: approve, reject, apply suggested fix and rerun that
  object only.
- Idempotent reruns keyed by scene version + object + fix.
- A footer with time-to-usable-scene, manual fixes, reruns, simulated cost and
  cost saved by idempotency.

Out (listed in the README as next steps):

- USD input, zip uploads with mesh files, `<include>`.
- Persistence: all state is in memory and is lost on restart.
- Auth and multi-user isolation beyond an unguessable scene id.
- A real pricing backend; cost is a published, fake rate card.
- End-to-end browser tests.

## Definitions

- **Object**: a MuJoCo body other than the world body, together with the geoms
  attached directly to it. Object identity is the body name; unnamed bodies get
  a stable generated name (`body_<index>`) at load time.
- **Finding**: `{object, rule, severity: "error"|"warning", message, fix?}`.
  `fix` is an optional `{fix_id, label}` describing a suggested change.
- **Movable object**: an object that has a joint on itself or on any ancestor
  body. Mass and friction rules apply only to movable objects; fixed scenery
  such as the table is exempt.
- **Scene version**: SHA-256 of the current MJCF XML text. Applying a fix
  produces a new XML text and therefore a new version.

## Rules

Static (stage 2):

| rule | fires when | fix |
| -- | -- | -- |
| `zero_mass` | a movable object's mass ≤ 1e-6 kg | set geom density to 1000 kg/m³ and recompute inertia |
| `bad_inertia` | a principal inertia ≤ 0 or violates the triangle inequality | recompute inertia from geoms |
| `no_collision` | every geom on the object has `contype == 0 and conaffinity == 0` | set `contype=1, conaffinity=1` |
| `default_friction` | friction equals MuJoCo's default `(1, 0.005, 0.0001)` on a movable object | warning only, no fix |
| `unlimited_hinge` | a hinge or slide joint has `limited` false | add a range of ±90° (hinge) or ±0.5 m (slide) |
| `implausible_scale` | the object's bounding box is larger than 20 m or smaller than 5 mm | when the box is larger than 20 m and dividing by 1000 lands between 5 mm and 20 m, scale geom sizes and positions by 0.001; otherwise warning only |

Simulation (stage 3), after `mj_forward` and stepping to 2 s:

| rule | fires when |
| -- | -- |
| `initial_penetration` | at t = 0 a contact involving the object has distance < −1 mm |
| `exploded` | any linear speed of the object exceeds 20 m/s |
| `fell_through_floor` | the object's lowest point ends more than 0.5 m below the floor plane |
| `unstable` | any `qpos`/`qvel` becomes NaN or MuJoCo raises an instability warning |

Simulation findings carry no automatic fix; the message names the likely cause.

An exception inside one rule produces a `check_error` warning for that rule
and the run continues.

## Sample scene

A hand-written tabletop MJCF (floor, table, and the objects below), written
from scratch for this project:

| object | planted issue |
| -- | -- |
| `mug_a` | zero mass |
| `plate` | collision disabled |
| `box_small` | starts 2 cm inside the table top |
| `cabinet_door` | hinge without limits |
| `tray_mm` | modelled in millimetres (size 300 instead of 0.3) |
| `mug_b` | none — the control object |

## Architecture

Single service. FastAPI serves the API and the built frontend from one origin,
deployed as one Docker image on Render.

### Backend (Python 3.12, FastAPI, mujoco 3.x)

| module | responsibility |
| -- | -- |
| `loader.py` | validate upload, compile with `MjSpec`, build the object list |
| `checks/static_rules.py` | static rules, each a pure function `(model, data) -> list[Finding]` |
| `checks/sim_rules.py` | simulation stepping and the simulation rules |
| `fixes.py` | apply a fix through `MjSpec` edits and return new XML |
| `geometry.py` | export compiled geometry for rendering |
| `runs.py` | run records, event buffers, idempotency index, pricing, metrics |
| `api.py` | routes; CPU-bound work runs in `asyncio.to_thread` |

### API

| method and path | result |
| -- | -- |
| `GET /api/samples` | list of built-in samples |
| `POST /api/scenes` | body: XML file upload or `{"sample": "tabletop"}` → `{scene_id, version}` |
| `POST /api/scenes/{scene_id}/runs` | start a full run → `{run_id}` |
| `GET /api/runs/{run_id}/events` | SSE stream; honours `Last-Event-ID` |
| `POST /api/scenes/{scene_id}/objects/{name}/fix` | body `{fix_id}` → `{run_id, cached}` |
| `POST /api/scenes/{scene_id}/objects/{name}/review` | body `{decision: "approve"|"reject"}` → object state |
| `GET /api/scenes/{scene_id}` | snapshot: objects, findings, decisions, geometry, metrics |

Errors use one shape: `{"error": {"code", "message"}}` with a matching HTTP
status.

### Events

Each event has a monotonically increasing integer `id` per run and a JSON
`data` payload:

- `stage_started {stage}`
- `geometry {objects: [...]}` — after compile
- `finding {finding}`
- `object_status {object, status}` — `checking | ok | has_issues`
- `stage_completed {stage, duration_ms, cost}`
- `run_completed {metrics}`
- `run_failed {stage, code, message}`

A run keeps every event it emitted. A client reconnecting with
`Last-Event-ID: n` receives events `n+1..` and then live events. A completed
run replays its buffer and closes.

### Idempotency and cost

Fix-and-rerun key: `sha256(scene_version + object + fix_id)`. If the key has a
run, the endpoint returns that `run_id` with `cached: true`, charges nothing,
and adds the would-be cost to "saved by idempotency". A rerun only executes
stages 2–3 for the one object, against the fixed scene, but the fixed XML
becomes the scene's current version.

Rate card (fake, shown in the UI and README): compile $0.001 per run, static
checks $0.0005 per object, simulation $0.01 per simulated second.

### Metrics

- **time to usable scene**: from scene creation to the moment every object is
  approved; shown as "—" until then.
- **manual fixes**: fix requests that were not cached.
- **reruns**: object reruns executed.
- **cost** and **saved by idempotency**.

### Frontend (Vite, React, TypeScript, React Three Fiber)

- `reduceEvent(state, event)` — a pure reducer from events to view state.
- `StageProgress` — the three stages with status and duration.
- `SceneViewport` — renders boxes, spheres, capsules, cylinders, planes and
  mesh geoms; objects with errors are red, warnings amber, approved green;
  clicking selects.
- `ObjectPanel` — findings for the selected object and the three actions.
- `MetricsFooter` — the metrics above.

Selection is shared between the viewport and the panel.

## Security and limits

- Uploads: XML only, ≤ 1 MB.
- Rejected before compile: `<include>`, and any `file` attribute
  (meshes, textures, heightfields) — the server never reads a path named by an
  upload.
- Simulation: at most 2 s simulated time and 10 s wall time per run.
- Scenes expire from memory after 2 hours; at most 200 scenes are kept.

## Testing

- pytest: one tiny MJCF fixture per rule (fires and does not fire), the
  upload validator, fix application, idempotency (second request is cached and
  free), and SSE replay from `Last-Event-ID`.
- Vitest: the event reducer.
- Manual: one full pass on the deployed URL against the success criteria.

## Deployment

Multi-stage Dockerfile (Node build → `python:3.12-slim`), one Render web
service on the free plan. The repository is public on the author's personal
GitHub account. Cold starts of ~30 s on the free plan are noted in the README.
