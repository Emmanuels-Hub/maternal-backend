"""Health-check blueprint — /health"""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "maternal-backend"}), 200
