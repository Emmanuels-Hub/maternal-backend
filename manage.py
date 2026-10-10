"""
Maternal Backend — entry point.

Run locally:
    python manage.py

Run with Gunicorn (production):
    gunicorn "manage:app" --bind 0.0.0.0:5000 --workers 1

Environment variables (see app/config.py for the full list):
  FLASK_DEBUG=true           — enable debug / hot-reload
  ENABLE_LOCAL_MODEL=false   — keep this FALSE on your dev machine;
                               only set to TRUE on an inference server
                               that already has the model weights.
  NATLAS_MODEL_PATH=...      — path to the downloaded model directory
  PORT=5000                  — port to bind
"""

import logging
import os

from app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug)
