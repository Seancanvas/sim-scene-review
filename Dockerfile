FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt ./
# Without this, `import mujoco` tries to load GLFW, which needs libX11 the slim image lacks.
ENV MUJOCO_GL=disable
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import mujoco, fastapi; print(mujoco.__version__)"
COPY backend/ ./
COPY --from=web /web/dist ./frontend_dist
ENV FRONTEND_DIST=/app/frontend_dist
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
