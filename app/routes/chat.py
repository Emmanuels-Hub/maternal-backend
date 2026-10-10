"""
POST /api/v1/chat

Health chatbot endpoint.

Body:
{
  "messages": [
    {"role": "user", "content": "What is a normal blood pressure?"}
  ],
  "user_text": "What is a normal blood pressure?",
  "image_base64": null   // optional base64 image string
}
"""

import logging
from flask import Blueprint, current_app, jsonify, request

from app.services.model_service import get_model_handler

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)


@chat_bp.post("/chat")
def chat():
    body = request.get_json(silent=True) or {}
    messages: list = body.get("messages", [])
    user_text: str = body.get("user_text", "")
    image_base64: str | None = body.get("image_base64")

    if not user_text:
        return jsonify({"error": "user_text is required"}), 400

    logger.info("[chat] user_text='%s...'", user_text[:60])

    try:
        handler = get_model_handler(current_app)
        reply = handler.generate_chat_response(
            messages=messages,
            user_text=user_text,
            image_base64=image_base64,
        )
        return jsonify({"reply": reply, "model_loaded": handler.is_loaded})
    except Exception as exc:
        logger.exception("[chat] Error: %s", exc)
        return jsonify({"error": str(exc)}), 500
