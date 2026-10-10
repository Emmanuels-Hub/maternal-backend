"""
Application configuration.

Environment variables (set in a .env file or your server environment):
    ENABLE_LOCAL_MODEL   — "true" to load the N-ATLaS GGUF model
    ENABLE_ASR_MODEL     — "true" to enable language-specific ASR (defaults to ENABLE_LOCAL_MODEL)
    HF_TOKEN             — accepted Hugging Face token for gated N-ATLaS ASR repositories
    NATLAS_ASR_DEVICE    — "auto", "cpu", or "cuda"
    NATLAS_MODEL_PATH    — optional absolute path to a local GGUF file
    NATLAS_MODEL_ID      — Hugging Face repository containing the GGUF
    NATLAS_MODEL_REVISION — pinned Hugging Face commit
    NATLAS_GPU_LAYERS    — llama.cpp layers to offload; -1 means all layers
  FLASK_SECRET_KEY     — secret key for Flask sessions
  PORT                 — port to bind (default 5000)
"""

import os


class Config:
    SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "natlas-dev-secret-key")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1")

    # ── N-ATLaS GGUF settings ──────────────────────────────────────────────
    # IMPORTANT: ENABLE_LOCAL_MODEL defaults to "false" so that running this
    # project on a laptop NEVER triggers a multi-GB Hugging Face download.
    # Enable it only on a machine intended to run model inference.
    ENABLE_LOCAL_MODEL: bool = os.getenv("ENABLE_LOCAL_MODEL", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    ENABLE_ASR_MODEL: bool = os.getenv(
        "ENABLE_ASR_MODEL", os.getenv("ENABLE_LOCAL_MODEL", "false")
    ).lower() in ("true", "1", "yes")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    NATLAS_ASR_DEVICE: str = os.getenv("NATLAS_ASR_DEVICE", "auto")
    NATLAS_MODEL_ID: str = os.getenv(
        "NATLAS_MODEL_ID", "tosinamuda/N-ATLaS-GGUF"
    )
    NATLAS_MODEL_REVISION: str = os.getenv(
        "NATLAS_MODEL_REVISION", "af9195261dd163807b31318b9b9d577de5ae3b7e"
    )
    NATLAS_GGUF_FILENAME: str = os.getenv(
        "NATLAS_GGUF_FILENAME", "N-ATLaS-GGUF-Q4_K_M.gguf"
    )
    NATLAS_MODEL_PATH: str = os.getenv("NATLAS_MODEL_PATH", "")
    NATLAS_CONTEXT_SIZE: int = int(os.getenv("NATLAS_CONTEXT_SIZE", "4096"))
    NATLAS_GPU_LAYERS: int = int(os.getenv("NATLAS_GPU_LAYERS", "0"))
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

    PORT: int = int(os.getenv("PORT", "5000"))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
