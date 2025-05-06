import logging
from TTS.api import TTS
import os
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_tts_models():
    """Download required TTS models."""
    models = {
        "en": "tts_models/en/ljspeech/tacotron2-DDC",
        "pl": "tts_models/pl/mai_female/vits"
    }
    
    for lang, model in models.items():
        try:
            logger.info(f"Downloading {lang} model: {model}")
            tts = TTS(model)
            logger.info(f"Successfully downloaded {lang} model")
        except Exception as e:
            logger.error(f"Error downloading {lang} model: {str(e)}")
            raise

def download_piper_model(model_name: str):
    """Download and setup a Piper TTS model."""
    model_dir = os.path.join("models", model_name)
    if os.path.exists(model_dir):
        logger.info(f"Model directory already exists: {model_dir}")
        return
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Created model directory: {model_dir}")
    
    # Parse model name to get language and voice details
    # Format: en_US-hfc_female-medium
    parts = model_name.split('-')
    if len(parts) != 3:
        raise Exception(f"Invalid model name format: {model_name}")
    
    lang_code, voice, quality = parts
    lang = lang_code.split('_')[0]  # Extract language code (e.g., 'en' from 'en_US')
    
    # Download model files
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    files = {
        f"{model_name}.onnx.json": f"{base_url}/{lang}/{lang_code}/{voice}/{quality}/{model_name}.onnx.json",
        f"{model_name}.onnx": f"{base_url}/{lang}/{lang_code}/{voice}/{quality}/{model_name}.onnx",
        "MODEL_CARD": f"{base_url}/{lang}/{lang_code}/{voice}/{quality}/MODEL_CARD"
    }
    
    for file, url in files.items():
        try:
            logger.info(f"Downloading {file} from {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                file_path = os.path.join(model_dir, file)
                with open(file_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Successfully downloaded {file} to {file_path}")
            else:
                logger.error(f"Failed to download {file} for model {model_name}. Status code: {response.status_code}")
                raise Exception(f"Failed to download {file} for model {model_name}. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {file} for model {model_name}: {str(e)}")
            raise Exception(f"Error downloading {file} for model {model_name}: {str(e)}")
    
    logger.info(f"Successfully downloaded all files for model {model_name}")

def download_piper_models():
    """Download all required Piper TTS models."""
    models = {
        "en": "en_US-hfc_female-medium",
        "pl": "pl_PL-gosia-medium"
    }
    
    for model in models.values():
        try:
            logger.info(f"Downloading Piper model: {model}")
            download_piper_model(model)
            logger.info(f"Successfully downloaded Piper model: {model}")
        except Exception as e:
            logger.error(f"Error downloading Piper model {model}: {str(e)}")
            raise

def download_all_models():
    """Download all required TTS models."""
    logger.info("Starting model download process")
    download_tts_models()
    download_piper_models()
    logger.info("Model download process completed")

if __name__ == "__main__":
    download_all_models() 