"""
Maternal Backend — Flask application factory.

The N-Atlas 4-bit quantised model is NOT downloaded here.
Set ENABLE_LOCAL_MODEL=true + NATLAS_MODEL_PATH=/path/to/model
only on the inference server that already has the weights.
"""

from flask import Flask
from flask_cors import CORS

from app.config import Config


def create_app(config_class: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Allow cross-origin requests from the Flutter app (mobile or web builds)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Register blueprints ────────────────────────────────────────────────
    from app.routes.health import health_bp
    from app.routes.analysis import analysis_bp
    from app.routes.chat import chat_bp
    from app.routes.asr import asr_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(analysis_bp, url_prefix="/api/v1")
    app.register_blueprint(chat_bp, url_prefix="/api/v1")
    app.register_blueprint(asr_bp, url_prefix="/api/v1")

    return app
