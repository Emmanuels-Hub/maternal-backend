"""
Model service — lazy-initialises the NAtlasModelHandler singleton
and attaches it to the Flask app context.

The handler is created once per process (stored on app.extensions).
ENABLE_LOCAL_MODEL=false (the default) means only the heuristic fallback
runs; no Hugging Face download is triggered.
"""

from __future__ import annotations

import logging
from flask import Flask

from n_atlas_model import NAtlasModelHandler

logger = logging.getLogger(__name__)

_EXTENSION_KEY = "natlas_model_handler"


def get_model_handler(app: Flask) -> NAtlasModelHandler:
    """
    Returns the singleton NAtlasModelHandler for this Flask app.
    Initialises it on first call.
    """
    if _EXTENSION_KEY not in app.extensions:
        logger.info(
            "[ModelService] Initialising NAtlasModelHandler "
            "(ENABLE_LOCAL_MODEL=%s, path='%s')",
            app.config.get("ENABLE_LOCAL_MODEL"),
            app.config.get("NATLAS_MODEL_PATH", ""),
        )
        handler = NAtlasModelHandler(
            model_id=app.config.get("NATLAS_MODEL_ID", "tosinamuda/N-ATLaS-GGUF"),
            model_path=app.config.get("NATLAS_MODEL_PATH", ""),
            enabled=app.config.get("ENABLE_LOCAL_MODEL", False),
            model_revision=app.config.get("NATLAS_MODEL_REVISION"),
            gguf_filename=app.config.get("NATLAS_GGUF_FILENAME"),
            context_size=app.config.get("NATLAS_CONTEXT_SIZE", 4096),
            gpu_layers=app.config.get("NATLAS_GPU_LAYERS", 0),
        )
        app.extensions[_EXTENSION_KEY] = handler

    return app.extensions[_EXTENSION_KEY]
