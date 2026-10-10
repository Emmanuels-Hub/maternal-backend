"""POST /api/v1/asr: transcribe audio with one selected N-ATLaS ASR model."""

import logging
import os
import tempfile

from flask import Blueprint, current_app, jsonify, request

from app.services.asr_model_service import (
    ASRDisabledError,
    UnsupportedASRLanguageError,
    get_asr_handler,
)

logger = logging.getLogger(__name__)

asr_bp = Blueprint("asr", __name__)
_SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


@asr_bp.post("/asr")
def asr():
    if "audio_file" not in request.files:
        return jsonify({"error": "audio_file is required"}), 400

    audio_file = request.files["audio_file"]
    language = request.form.get("language", "english").strip().lower()
    tmp_path: str | None = None

    suffix = os.path.splitext(audio_file.filename or "audio.wav")[1].lower()
    if suffix not in _SUPPORTED_AUDIO_EXTENSIONS:
        return jsonify({"error": "Unsupported audio file type"}), 400

    logger.info("[asr] Received audio for language=%s", language)
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        transcript = get_asr_handler(current_app).transcribe(tmp_path, language)
        return jsonify({"transcript": transcript, "language": language})
    except UnsupportedASRLanguageError as exc:
        return jsonify({"error": str(exc)}), 422
    except ASRDisabledError:
        return jsonify({"error": "ASR inference is disabled on this server"}), 503
    except Exception:
        logger.exception("[asr] Inference failed for language=%s", language)
        return jsonify({"error": "ASR inference failed"}), 503
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning("[asr] Could not remove temporary audio file")
