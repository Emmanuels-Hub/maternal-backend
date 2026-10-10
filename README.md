# Maternal N-ATLaS API

Flask API for PatDoc scan explanations. Optional N-ATLaS GGUF and ASR models
can be served from mounted Cloud Storage model files on Cloud Run.

For Cloud Run deployment with model files mounted from Cloud Storage, see
[CLOUD_RUN.md](CLOUD_RUN.md).

The API accepts demo scan readings at `/api/v1/analyze` and exposes `/health`
for a health check. Do not send real patient information to this unauthenticated
demo deployment.