import sys
from pathlib import Path

# Add project root to python path so imports resolve properly on serverless environments
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from app.backend.main import app
