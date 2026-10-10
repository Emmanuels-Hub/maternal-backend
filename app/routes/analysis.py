"""
POST /api/v1/analyze

Receives sensor data + patient info from the Flutter app and returns:
  - risk_level  : "Low Risk" / "Mid Risk" / "High Risk"
  - explanation : AI doctor note (from N-Atlas or heuristic fallback)
  - language    : echo of the requested language

Expected JSON body:
{
  "sensor_data": {
    "bloodPressureSystolic": 120,
    "bloodPressureDiastolic": 80,
    "heartRate": 72,
    "bodyTemp": 36.6,
    "spo2": 98,
    "ecgResult": "Normal",
    "ecgVoltage": 0.45,
    "weight": 65.0
  },
  "risk_level": "Low Risk",          // result from on-device TFLite model
  "language": "English",             // target language for the note
  "patient_info": {                  // optional
    "fullName": "Jane Doe",
    "age": 28,
    ...
  }
}
"""

import logging
from flask import Blueprint, current_app, jsonify, request

from app.services.model_service import get_model_handler

logger = logging.getLogger(__name__)

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.post("/analyze")
def analyze():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    sensor_data = body.get("sensor_data")
    if not sensor_data:
        return jsonify({"error": "sensor_data is required"}), 400

    risk_level: str = body.get("risk_level", "Low Risk")
    language: str = body.get("language", "English")
    patient_info: dict | None = body.get("patient_info")

    logger.info(
        "[analyze] risk=%s lang=%s patient=%s",
        risk_level,
        language,
        patient_info.get("fullName") if patient_info else "N/A",
    )

    try:
        handler = get_model_handler(current_app)
        explanation = handler.generate_doctor_explanation(
            sensor_data=sensor_data,
            risk_level=risk_level,
            patient_info=patient_info,
            language=language,
        )
        return jsonify(
            {
                "risk_level": risk_level,
                "explanation": explanation,
                "language": language,
                "model_loaded": handler.is_loaded,
            }
        )
    except Exception as exc:
        logger.exception("[analyze] Unexpected error: %s", exc)
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500


@analysis_bp.post("/translate")
def translate():
    """
    POST /api/v1/translate
    Body: { "text": "...", "target_language": "Yoruba" }
    """
    body = request.get_json(silent=True) or {}
    text: str = body.get("text", "")
    target_language: str = body.get("target_language", "English")

    if not text:
        return jsonify({"error": "text is required"}), 400

    try:
        handler = get_model_handler(current_app)
        translated = handler.translate_text(text, target_language)
        return jsonify({"translated": translated, "target_language": target_language})
    except Exception as exc:
        logger.exception("[translate] Error: %s", exc)
        return jsonify({"error": str(exc)}), 500
