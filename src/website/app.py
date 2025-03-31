from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import asyncio
import json
import sys
import os
from typing import List, Dict, Any
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.route("/", methods=["GET"])
def index():
    """Serve the HTML interface (for simple deployments)"""
    with open("src/website/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    # Run the Flask app with asyncio integration
    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.bind = ["0.0.0.0:8080"]

    asyncio.run(serve(app, config))
