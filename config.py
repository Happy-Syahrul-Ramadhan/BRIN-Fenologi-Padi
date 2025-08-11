import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class untuk mengambil environment variables"""
    
    # Google Earth Engine Configuration
    GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID')
    if not GEE_PROJECT_ID:
        raise ValueError("GEE_PROJECT_ID is missing in the environment variables.")
    
    # Model Configuration  
    MODEL_PATH = os.getenv('MODEL_PATH')
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH is missing in the environment variables.")
    
    # Earth Engine Assets (tanpa fallback default)
    INDRAMAYU_AOI_ASSET = os.getenv('INDRAMAYU_AOI_ASSET')
    if not INDRAMAYU_AOI_ASSET:
        raise ValueError("INDRAMAYU_AOI_ASSET is missing in the environment variables.")
    
    TRAINING_POINTS_ASSET = os.getenv('TRAINING_POINTS_ASSET')
    if not TRAINING_POINTS_ASSET:
        raise ValueError("TRAINING_POINTS_ASSET is missing in the environment variables.")
    
    COLLECTION_ASSET = os.getenv('COLLECTION_ASSET')
    if not COLLECTION_ASSET:
        raise ValueError("COLLECTION_ASSET is missing in the environment variables.")
    
    # Processing Parameters
    SCALE = int(os.getenv('SCALE'))
    if not SCALE:
        raise ValueError("SCALE is missing in the environment variables.")
    
    MAX_TRAINING_POINTS = int(os.getenv('MAX_TRAINING_POINTS'))
    if not MAX_TRAINING_POINTS:
        raise ValueError("MAX_TRAINING_POINTS is missing in the environment variables.")
    
    # Flask Configuration (optional)
    FLASK_ENV = os.getenv('FLASK_ENV')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL')

# Instance konfigurasi yang bisa diimpor
config = Config()
