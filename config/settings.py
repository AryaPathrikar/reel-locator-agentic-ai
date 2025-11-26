from pathlib import Path
from dotenv import load_dotenv
import os

# Base directory of the project (reel_locator/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env at project root
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# API keys (fill these in .env)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# Data directories
DATA_DIR = BASE_DIR / "data"
FRAMES_DIR = DATA_DIR / "frames"

# Ensure frames dir exists
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
