# Architecture and flows

Diagrams for the paths a scene takes through Sim-Scene Review. They render on
GitHub (Mermaid); the prose version of the same decisions is in the
[README](../README.md) and the [design spec](specs/2026-10-15-sim-scene-review-design.md).

## 1. Components

```mermaid
flowchart LR
    subgraph Browser["Browser (Vite + React + TypeScript)"]
        UI[SceneInput · StageProgress · ObjectPanel · MetricsFooter]
        R3F[SceneViewport<br/>React Three Fiber]
        RED[reducer.ts<br/>pure view-state fold]
        ES[useRunStream<br/>EventSource]
        UI --> RED
        R3F --> RED
        ES --> RED
    end

    subgraph Service["One FastAPI process (python:3.12-slim image)"]
        API[api.py<br/>async handlers]
        REG[runs.py · Registry<br/>200 scenes · 2 h TTL · 20 runs per scene]
        PIPE[pipeline.py<br/>execute in a worker thread]
        LOAD[loader.py<br/>validate · canonicalise · compile]
        STAT[checks/static_rules.py]
        SIM[checks/sim_rules.py]
        FIX[fixes.py · MjSpec edits]
        GEO[geometry.py]
        MJ[(MuJoCo 3.3.7)]
        STATIC[dist · index.html · /assets]
    end

    UI -- "fetch /api/*" --> API
    ES -- "GET /api/runs/{id}/events (SSE)" --> API
    API --> REG
    API -- "asyncio.to_thread" --> PIPE
    API -- "asyncio.to_thread" --> LOAD
    API -- "asyncio.to_thread" --> FIX
    PIPE --> LOAD --> MJ
    PIPE --> GEO --> MJ
    PIPE --> STAT --> MJ
    PIPE --> SIM --> MJ
    PIPE -- "emit(event)" --> REG
    FIX --> MJ
    Browser -. "GET / and /assets" .-> STATIC
```

Everything the browser draws comes from the compiled `MjModel`
(`geometry.py`), so the viewport shows exactly what the simulation stage
checks.

## 2. A review run

`POST /api/scenes` (sample) or `POST /api/scenes/upload` creates a scene;
`POST /api/scenes/{id}/runs` starts a full run. The pipeline runs in a worker
thread and writes events into the run's buffer; the browser reads them over
SSE (section 4).

```mermaid
flowchart TD
    A[MJCF text arrives] --> V{"validate_xml<br/>≤ 1 MB · no include · no file= · root mujoco"}
    V -- fail --> E400[400 invalid_mjcf / too_large]
    V -- ok --> C["canonicalise<br/>name unnamed bodies, re-serialise via MjSpec"]
    C --> K{"compile_scene<br/>MjModel + mj_forward"}
    K -- MuJoCo error --> E400c[400 compile_error]
    K -- "nbody > 500 · ngeom > 2000<br/>nmeshvert > 200k · narena > 64 MiB" --> E413[400 scene_too_large]
    K -- ok --> S["Registry.add_scene<br/>version = sha256 of XML<br/>objects: name · body_id · movable · geom_ids"]
    S --> R["POST /runs → 202 run_id<br/>409 run_in_progress if one is active"]

    R --> L1[stage_started load]
    L1 --> L2["compile again from scene.xml<br/>export geometry first: simulate mutates MjData"]
    L2 --> L3["geometry · object_status checking per target<br/>stage_completed load · cost 0.001"]
    L3 --> ST1[stage_started static]
    ST1 --> ST2["run_static_rules per object<br/>zero_mass · no_collision · default_friction<br/>unlimited_joint · implausible_scale<br/>a rule that throws becomes a check_error warning"]
    ST2 --> ST3["finding × N<br/>stage_completed static · cost 0.0005 × objects"]
    ST3 --> SM1[stage_started simulation]
    SM1 --> SM2["simulate 2 s sim / 10 s wall<br/>initial_penetration at t=0<br/>per step: MuJoCo warning counters → unstable<br/>speed > 20 m/s → exploded<br/>after: lowest AABB corner 0.5 m below floor → fell_through_floor"]
    SM2 --> SM3["finding × M<br/>stage_completed simulation · cost 0.01 × simulated s"]
    SM3 --> D["per object: has_issues if findings else ok<br/>object_status × targets"]
    D --> RC[run_completed · metrics]

    L2 -- SceneError --> RF[run_failed stage=load]
    ST2 -- unexpected exception --> RF2["restore saved status and findings per target<br/>run_failed stage=internal · generic message · server log"]
    SM2 -- unexpected exception --> RF2
```

## 3. Fix, rerun and idempotency

`POST /api/scenes/{id}/objects/{name}/fix {fix_id}`. A fix edits the scene
through `MjSpec`, the result is compiled fresh, and only that object is
re-checked (`only=name`). The same fix on the same version is answered from
the cache without compute.

```mermaid
flowchart TD
    F[POST objects/name/fix] --> L1{{lock scene}}
    L1 --> O{object exists?}
    O -- no --> E404[404 object_not_found]
    O -- yes --> KEY["key = sha256(version : object : fix_id)"]
    KEY --> HIT{scene.fix_runs has key?}
    HIT -- "yes, run exists and did not fail" --> CACHED["saved += rerun_cost<br/>202 run_id = previous, cached = true"]
    HIT -- "yes, but that run failed or was evicted" --> DROP[forget every key pointing at it] --> ACT
    HIT -- no --> ACT{run in progress?}
    ACT -- yes --> E409[409 run_in_progress]
    ACT -- no --> SNAP["remember base_xml, base_version<br/>release lock"]
    SNAP --> AP["apply_fix base_xml, name, fix_id<br/>in a worker thread, outside the lock"]
    AP -- "unknown fix or object" --> E400[400 unknown_fix · 404 object_not_found]
    AP -- new XML --> L2{{lock scene again}}
    L2 --> CHK{"version unchanged and<br/>still no active run?"}
    CHK -- no --> E409b[409 scene changed while the fix was prepared]
    CHK -- yes --> REC["add_run scene, name<br/>fix_runs key of base_version = run<br/>fix_runs key of new version = run<br/>manual_fixes += 1"]
    REC --> RUN["execute scene, run, xml=new, only=name<br/>202 run_id, cached = false"]
    RUN --> OK{compiled?}
    OK -- yes --> SWAP["scene.xml · version · objects · geometry replaced<br/>only that object's findings and decision reset"]
    OK -- target missing --> MISS["run_failed object_missing_after_fix<br/>scene untouched"]
    OK -- compile error --> KEEP["run_failed stage=load<br/>scene untouched · not cached"]
```

The four fixes:

| fix_id | MjSpec edit | Offered by |
| -- | -- | -- |
| `recompute_mass` | clear explicit inertia, density 1000, mass unset | `zero_mass` |
| `enable_collision` | contype = conaffinity = 1 on every geom | `no_collision` |
| `limit_joints` | hinge ±90°, slide ±0.5 m on joints without a range | `unlimited_joint` |
| `scale_mm_to_m` | geom size and pos × 0.001 | `implausible_scale` (only when ÷1000 lands in range) |

## 4. Streaming a run to the browser

Each run keeps its events in memory with increasing ids. The server replays
from `Last-Event-ID` and then blocks in `Condition.wait_for` on a worker
thread; the browser's `EventSource` reconnects on its own.

```mermaid
sequenceDiagram
    participant B as Browser
    participant A as api.py (event loop)
    participant W as pipeline worker thread
    participant R as RunRecord buffer

    B->>A: POST /scenes/{id}/runs
    A->>W: asyncio.to_thread(execute)
    A-->>B: 202 {run_id}
    B->>A: GET /runs/{run_id}/events  (Last-Event-ID: 0)
    loop every emit
        W->>R: emit(event): id = n+1, notify_all
        R-->>A: wait_for(last_id) returns
        A-->>B: id: n+1 / event: name / data: json
    end
    Note over A,B: idle 15 s → ": keepalive"
    B--xA: connection drops
    B->>A: GET /runs/{run_id}/events  (Last-Event-ID: k)
    A-->>B: replay ids k+1 … n, then keep waiting
    W->>R: emit(run_completed)
    A-->>B: run_completed, stream closes
    B->>A: GET /scenes/{id}  (snapshot: decisions, version, metrics, current_run_done)
    A-->>B: authoritative snapshot → reducer.fromSnapshot
```

Events the browser folds: `stage_started`, `stage_completed`, `geometry`,
`object_status`, `finding`, `run_completed`, `run_failed`.

## 5. Object lifecycle

```mermaid
stateDiagram-v2
    [*] --> pending: scene created
    pending --> checking: run starts (object_status)
    checking --> ok: no findings
    checking --> has_issues: one or more findings
    has_issues --> checking: fix, rerun of this object only
    ok --> checking: Run again (full pass)
    has_issues --> checking: Run again
    checking --> has_issues: internal failure restores the saved state
    checking --> ok: internal failure restores the saved state

    state decision {
        [*] --> none
        none --> approved: POST review approve
        none --> rejected: POST review reject
        approved --> rejected
        rejected --> approved
        approved --> none: object re-checked after a fix
        rejected --> none: object re-checked after a fix
    }
```

`status` and `decision` are independent fields on `ObjectState`. The scene
is marked usable the first time every object is `approved`; that moment is
the **time to usable scene** metric.

## 6. Frontend state

```mermaid
flowchart LR
    SNAP[GET /scenes/id snapshot] --> FS["fromSnapshot<br/>running = current_run_id set and not current_run_done"]
    EV[SSE event] --> RE[reduceEvent]
    CLICK[Fix button] --> BR[beginRun + setPendingFix]
    RC[run_completed, refetch snapshot] --> RAF["recordAppliedFix with snap.version<br/>clear pendingFix"]
    RFL[run_failed] --> CLR[clear pendingFix]
    REV[Approve / Reject response] --> AR[applyReview]
    FS & RE & BR & RAF & CLR & AR --> VS[(ViewState)]
    VS --> UI[ObjectPanel · StageProgress · MetricsFooter]
    VS --> VP["SceneViewport colours<br/>approved green · rejected grey · checking blue<br/>error red · warning orange · else geom rgba"]
    UI -- "Re-apply only when entry.version equals state.version" --> CLICK
```

All reducer functions are pure and unit-tested (`frontend/src/reducer.test.ts`).
