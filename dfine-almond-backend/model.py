"""
D-FINE Almond Detection - Label Studio ML Backend
==================================================

Production-ready Docker deployment for D-FINE almond quality detection.

This module imports the main D-FINE backend from the parent directory
to avoid code duplication while maintaining a clean Docker structure.

Environment Variables:
    MODEL_PATH: Path to model_2.pt (default: /app/models/model_2.pt)
    LABEL_STUDIO_URL: Label Studio server URL
    LABEL_STUDIO_API_KEY: API key for Label Studio
    AWS_ACCESS_KEY_ID: AWS credentials for S3 access
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import main backend
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the main D-FINE backend class
try:
    from dfine_labelstudio_backend import DFineModel, MODEL_PATH, CLASS_NAMES
    logger.info("Successfully imported DFineModel from parent directory")
except ImportError as e:
    logger.error(f"Failed to import DFineModel: {e}")
    logger.error("Make sure dfine_labelstudio_backend.py exists in parent directory")
    raise

# Override model path from environment variable if provided
MODEL_PATH_ENV = os.getenv('MODEL_PATH', '/app/models/model_2.pt')
if MODEL_PATH_ENV != MODEL_PATH:
    logger.info(f"Overriding model path from environment: {MODEL_PATH_ENV}")
    # Monkey-patch the MODEL_PATH in the imported module
    import dfine_labelstudio_backend
    dfine_labelstudio_backend.MODEL_PATH = MODEL_PATH_ENV

# Re-export for label-studio-ml
__all__ = ['DFineModel']

# Print startup info
logger.info("="*70)
logger.info("D-FINE Almond Detection ML Backend")
logger.info("="*70)
logger.info(f"Model path: {MODEL_PATH_ENV}")
logger.info(f"Number of classes: {len(CLASS_NAMES)}")
logger.info(f"Label Studio URL: {os.getenv('LABEL_STUDIO_URL', 'not set')}")
logger.info("="*70)
