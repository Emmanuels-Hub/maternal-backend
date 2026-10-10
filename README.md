---
title: Maternal N-ATLaS API
sdk: docker
app_port: 7860
---

# Maternal N-ATLaS API

Flask API for PatDoc scan explanations. This Space configuration runs the
lightweight fallback mode on CPU Basic. It does not load the optional GGUF or
ASR models; ASR remains disabled unless separately configured with its model
dependencies and Hugging Face access.

The API accepts demo scan readings at `/api/v1/analyze` and exposes `/health`
for a health check. Do not send real patient information to this unauthenticated
demo deployment.