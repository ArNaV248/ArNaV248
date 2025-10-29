"""
WSGI Entry Point for D-FINE Label Studio ML Backend
====================================================

This module provides the WSGI application entry point for deployment
with production servers like Gunicorn.

Usage with Gunicorn:
    gunicorn -w 1 -b 0.0.0.0:9091 _wsgi:application

Note:
    - Use only 1 worker (-w 1) as the model is loaded in GPU memory
    - Multiple workers will cause CUDA errors
    - For multiple concurrent requests, increase threads per worker instead
"""

import os
import sys
from pathlib import Path

# Ensure parent directory is in path
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import the Label Studio ML backend framework
from label_studio_ml.api import init_app

# Import our model
from model import DFineModel

# Initialize the application
application = init_app(
    model_class=DFineModel,
    # Add any additional configuration here
)

# For backward compatibility
app = application

if __name__ == "__main__":
    # For development/testing only
    # In production, use: gunicorn _wsgi:application
    import uvicorn

    port = int(os.getenv("PORT", "9091"))
    uvicorn.run(
        "_ wsgi:application",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disable reload (model is large)
        workers=1,     # Single worker for GPU
        log_level="info"
    )
