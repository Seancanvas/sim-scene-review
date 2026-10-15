"""HTTP surface. Handlers are async so they can schedule pipeline work on the
loop; the CPU-bound pipeline itself runs in a worker thread."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import UploadFile

from . import pipeline
from .fixes import apply_fix
from .loader import MAX_UPLOAD_BYTES, SceneError, canonicalise, compile_scene, scene_version
from .runs import Registry, idempotency_key, rerun_cost
from .samples import SAMPLES, load_sample

router = APIRouter(prefix="/api")
KEEPALIVE_SECONDS = 15.0
# multipart framing and the field headers around the file itself
MULTIPART_OVERHEAD_BYTES = 200_000


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def registry(request: Request) -> Registry:
    return request.app.state.registry


def _scene(reg: Registry, scene_id: str):
    scene = reg.scene(scene_id)
    if scene is None:
        raise ApiError(404, "scene_not_found", f"no scene {scene_id}")
    return scene


def _active_run(reg: Registry, scene):
    run = reg.run(scene.current_run_id) if scene.current_run_id else None
    return run if run is not None and not run.done else None


def _start(request: Request, scene, run, **kwargs) -> None:
    task = asyncio.create_task(asyncio.to_thread(pipeline.execute, scene, run, **kwargs))
    tasks = request.app.state.tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


def _compile(text: str):
    return compile_scene(canonicalise(text))


async def _create(reg: Registry, text: str) -> dict:
    try:
        # compiling a large scene takes a while; keep the loop answering meanwhile
        compiled = await asyncio.to_thread(_compile, text)
    except SceneError as e:
        raise ApiError(400, e.code, e.message)
    scene = reg.add_scene(compiled.xml, compiled.version, compiled.objects)
    return {"scene_id": scene.scene_id, "version": scene.version}


class CreateScene(BaseModel):
    sample: str


class FixBody(BaseModel):
    fix_id: str


class ReviewBody(BaseModel):
    decision: Literal["approve", "reject"]


@router.get("/health")
async def health():
    return {"ok": True}


@router.get("/samples")
async def samples():
    return [{"id": key, "label": label} for key, label in SAMPLES.items()]


@router.post("/scenes", status_code=201)
async def create_from_sample(body: CreateScene, reg: Registry = Depends(registry)):
    if body.sample not in SAMPLES:
        raise ApiError(404, "sample_not_found", f"no sample {body.sample!r}")
    return await _create(reg, load_sample(body.sample))


@router.post("/scenes/upload", status_code=201)
async def create_from_upload(request: Request, reg: Registry = Depends(registry)):
    # The form is parsed here rather than through File(...): FastAPI reads a
    # declared body before the handler runs, which would spool an oversize
    # upload to disk before this header check could refuse it.
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
        except ValueError:
            too_big = False
        if too_big:
            raise ApiError(413, "too_large", "scene must be 1 MB or smaller")
    form = await request.form()
    file = form.get("file")
    if not isinstance(file, UploadFile):
        raise ApiError(422, "validation_error", "multipart field 'file' is required")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ApiError(413, "too_large", "scene must be 1 MB or smaller")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(400, "not_utf8", "upload must be UTF-8 encoded XML")
    return await _create(reg, text)


@router.get("/scenes/{scene_id}")
async def get_scene(scene_id: str, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        return scene.snapshot()


@router.post("/scenes/{scene_id}/runs", status_code=202)
async def start_run(scene_id: str, request: Request, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        if _active_run(reg, scene) is not None:
            raise ApiError(409, "run_in_progress", "wait for the current run to finish")
        run = reg.add_run(scene_id)
        scene.current_run_id, scene.current_run = run.run_id, run
    _start(request, scene, run)
    return {"run_id": run.run_id}


@router.post("/scenes/{scene_id}/objects/{name}/fix", status_code=202)
async def fix_object(scene_id: str, name: str, body: FixBody, request: Request, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        if name not in scene.objects:
            raise ApiError(404, "object_not_found", f"no object {name!r}")
        key = idempotency_key(scene.version, name, body.fix_id)
        cached = scene.fix_runs.get(key)
        if cached is not None:
            previous = reg.run(cached)
            if previous is None or previous.failed:
                # a failed run produced nothing worth reusing
                for stale in [k for k, v in scene.fix_runs.items() if v == cached]:
                    del scene.fix_runs[stale]
            else:
                scene.saved += rerun_cost()
                return {"run_id": cached, "cached": True}
        if _active_run(reg, scene) is not None:
            raise ApiError(409, "run_in_progress", "wait for the current run to finish")
        base_xml, base_version = scene.xml, scene.version
    try:
        # MjSpec work off the loop and outside the lock; the lock is re-taken below
        xml = await asyncio.to_thread(apply_fix, base_xml, name, body.fix_id)
    except SceneError as e:
        raise ApiError(400 if e.code == "unknown_fix" else 404, e.code, e.message)
    with scene.lock:
        if scene.version != base_version or _active_run(reg, scene) is not None:
            raise ApiError(409, "run_in_progress", "the scene changed while the fix was prepared; try again")
        run = reg.add_run(scene_id, name)
        # Asking for the same fix again on the scene this fix produces is the
        # same work, so both the input and the output version point at this run.
        for version in (base_version, scene_version(xml)):
            scene.fix_runs[idempotency_key(version, name, body.fix_id)] = run.run_id
        scene.manual_fixes += 1
        scene.current_run_id, scene.current_run = run.run_id, run
    _start(request, scene, run, xml=xml, only=name)
    return {"run_id": run.run_id, "cached": False}


@router.post("/scenes/{scene_id}/objects/{name}/review")
async def review_object(scene_id: str, name: str, body: ReviewBody, reg: Registry = Depends(registry)):
    scene = _scene(reg, scene_id)
    with scene.lock:
        state = scene.objects.get(name)
        if state is None:
            raise ApiError(404, "object_not_found", f"no object {name!r}")
        state.decision = "approved" if body.decision == "approve" else "rejected"
        scene.mark_usable_if_done()
        return {"object": state.to_dict(), "metrics": scene.metrics()}


async def _stream(run, last_id: int):
    while True:
        events = await asyncio.to_thread(run.wait_for, last_id, KEEPALIVE_SECONDS)
        for event in events:
            yield f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.data)}\n\n"
            last_id = event.id
        if run.done and last_id >= run.last_id:
            return
        if not events:
            yield ": keepalive\n\n"


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, reg: Registry = Depends(registry)):
    run = reg.run(run_id)
    if run is None:
        raise ApiError(404, "run_not_found", f"no run {run_id}")
    try:
        last_id = max(0, int(request.headers.get("last-event-id", "0") or 0))
    except ValueError:
        last_id = 0
    return StreamingResponse(
        _stream(run, last_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
