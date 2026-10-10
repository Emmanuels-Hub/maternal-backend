"""Lazy, single-model-at-a-time inference for the N-ATLaS ASR models."""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Any

from flask import Flask

logger = logging.getLogger(__name__)

ASR_MODELS = {
    "hausa": {
        "repo_id": "NCAIR1/Hausa-ASR",
        "revision": "e635b9eda29060c6114c8f4d8b2d903f5c83a44a",
        "language": "hausa",
    },
    "igbo": {
        "repo_id": "NCAIR1/Igbo-ASR",
        "revision": "180732299d5cba3dc8b289260ac84b7838bb3954",
        "language": "igbo",
    },
    "yoruba": {
        "repo_id": "NCAIR1/Yoruba-ASR",
        "revision": "d1ae7b8b79c2ccd547d8761effe5057433f3fc7f",
        "language": "yoruba",
    },
}

_LANGUAGE_ALIASES = {
    "ha": "hausa",
    "ibo": "igbo",
    "ig": "igbo",
    "yo": "yoruba",
}


class ASRDisabledError(RuntimeError):
    """Raised when ASR inference is disabled by configuration."""


class UnsupportedASRLanguageError(ValueError):
    """Raised when there is no configured N-ATLaS ASR model for a language."""


class ASRModelHandler:
    """Loads only the ASR model needed for the current request language."""

    def __init__(
        self,
        enabled: bool,
        hf_token: str = "",
        device: str = "auto",
        model_root: str = "",
    ) -> None:
        self.enabled = enabled
        self.hf_token = hf_token or None
        self.device_setting = device
        self.model_root = model_root
        self._pipeline: Any = None
        self._language: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded_language(self) -> str | None:
        return self._language

    def transcribe(self, audio_path: str, language: str) -> str:
        if not self.enabled:
            raise ASRDisabledError("ASR inference is disabled")

        normalized_language = language.strip().lower().replace("_", " ")
        normalized_language = _LANGUAGE_ALIASES.get(
            normalized_language, normalized_language
        )
        if normalized_language not in ASR_MODELS:
            supported = ", ".join(sorted(ASR_MODELS))
            raise UnsupportedASRLanguageError(
                f"No server ASR model for '{language}'. Supported languages: {supported}."
            )

        with self._lock:
            if self._language != normalized_language:
                self._load_model(normalized_language)

            result = self._pipeline(
                audio_path,
                chunk_length_s=30,
                stride_length_s=(5, 5),
                generate_kwargs={
                    "language": ASR_MODELS[normalized_language]["language"],
                    "task": "transcribe",
                },
            )

        transcript = result.get("text", "") if isinstance(result, dict) else ""
        return transcript.strip()

    def _load_model(self, language: str) -> None:
        import torch
        from transformers import pipeline

        self._release_model()
        device = self._resolve_device(torch)
        model_kwargs = {"torch_dtype": torch.float16} if device >= 0 else {}
        model_config = ASR_MODELS[language]
        model_path = (
            os.path.join(self.model_root, language) if self.model_root else None
        )

        if model_path and not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"ASR model directory for {language} was not found: {model_path}"
            )

        logger.info(
            "[N-ATLaS ASR] Loading %s from %s",
            language,
            model_path or f"{model_config['repo_id']}@{model_config['revision']}",
        )
        pipeline_options = {
            "device": device,
            "model_kwargs": model_kwargs,
        }
        if model_path:
            pipeline_options["model"] = model_path
        else:
            pipeline_options.update(
                {
                    "model": model_config["repo_id"],
                    "revision": model_config["revision"],
                    "token": self.hf_token,
                }
            )

        self._pipeline = pipeline("automatic-speech-recognition", **pipeline_options)
        self._language = language

    def _resolve_device(self, torch: Any) -> int:
        setting = self.device_setting.strip().lower()
        if setting == "auto":
            return 0 if torch.cuda.is_available() else -1
        if setting == "cpu":
            return -1
        if setting == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("NATLAS_ASR_DEVICE=cuda but CUDA is unavailable")
            return 0
        raise ValueError("NATLAS_ASR_DEVICE must be auto, cpu, or cuda")

    def _release_model(self) -> None:
        if self._pipeline is None:
            return

        self._pipeline = None
        self._language = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


_EXTENSION_KEY = "natlas_asr_handler"


def get_asr_handler(app: Flask) -> ASRModelHandler:
    """Return the per-process ASR handler attached to this Flask app."""
    if _EXTENSION_KEY not in app.extensions:
        app.extensions[_EXTENSION_KEY] = ASRModelHandler(
            enabled=app.config.get("ENABLE_ASR_MODEL", False),
            hf_token=app.config.get("HF_TOKEN", ""),
            device=app.config.get("NATLAS_ASR_DEVICE", "auto"),
            model_root=app.config.get("NATLAS_ASR_MODEL_ROOT", ""),
        )
    return app.extensions[_EXTENSION_KEY]