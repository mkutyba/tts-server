import os
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import logging
from datetime import datetime
from unidecode import unidecode
import threading
import queue
import time
import ctypes
from TTS.api import TTS
from pydub import AudioSegment
import re
import soundfile as sf
import subprocess
# from init import download_piper_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'tts-server_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not isinstance(exctype, type):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(tid),
        ctypes.py_object(exctype)
    )
    if res == 0:
        raise ValueError("Invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class TTSRequest(BaseModel):
    text: str
    language: str = "en"  # Default to English
    title: str = ""  # Optional title for the output file

class TTSManager:
    def __init__(self):
        self._stop_event = threading.Event()
        self._result_queue = queue.Queue()
        self._tts_thread = None
        self._last_activity = time.time()
        self._tts_models = {
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
            "pl": "tts_models/pl/mai_female/vits"
        }
        self._tts_instances = {}

    def get_tts(self, language: str):
        if language not in self._tts_instances:
            if language not in self._tts_models:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
            self._tts_instances[language] = TTS(self._tts_models[language])
        return self._tts_instances[language]

    def stop(self):
        self._stop_event.set()
        if self._tts_thread and self._tts_thread.is_alive():
            try:
                _async_raise(self._tts_thread.ident, SystemExit)
                self._tts_thread.join(timeout=1.0)
            except Exception as e:
                logger.error(f"Error stopping TTS thread: {str(e)}")
                # Don't try to force stop the thread as it may cause issues

    def update_activity(self):
        self._last_activity = time.time()

    def is_client_active(self, timeout=1.0):
        return time.time() - self._last_activity < timeout

    def generate_speech(self, text: str, language: str, output_path: str):
        try:
            self._tts_thread = threading.Thread(
                target=self._run_tts,
                args=(text, language, output_path)
            )
            self._tts_thread.daemon = True
            self._tts_thread.start()
            
            while not self._stop_event.is_set():
                try:
                    result = self._result_queue.get(timeout=0.1)
                    return result
                except queue.Empty:
                    if not self.is_client_active():
                        self._stop_event.set()
                        raise Exception("Client inactive")
                    continue
            
            raise Exception("TTS generation cancelled")
            
        except Exception as e:
            self._result_queue.put(e)
            raise

    def _run_tts(self, text: str, language: str, output_path: str):
        try:
            if not self._stop_event.is_set():
                tts = self.get_tts(language)
                logger.info("TTS instance created successfully")
                
                logger.info("Starting text synthesis...")
                # Use tts_to_file instead of synthesize
                tts.tts_to_file(text=text, file_path=output_path)
                logger.info("Text synthesis completed successfully")
                
                # Convert WAV to MP3
                logger.info("Converting WAV to MP3...")
                audio = AudioSegment.from_wav(output_path)
                mp3_path = output_path.replace('.wav', '.mp3')
                audio.export(mp3_path, format='mp3', bitrate='192k')
                logger.info(f"MP3 file saved successfully: {mp3_path}")
                
                # Remove the original WAV file
                logger.info("Removing temporary WAV file...")
                os.remove(output_path)
                logger.info("Temporary WAV file removed")
                
                if not self._stop_event.is_set():
                    self._result_queue.put(True)
                    logger.info("TTS generation completed successfully")
        except Exception as e:
            logger.error(f"Error in _run_tts: {str(e)}", exc_info=True)
            if not self._stop_event.is_set():
                self._result_queue.put(e)

class PiperTTSManager:
    def __init__(self):
        self._stop_event = threading.Event()
        self._result_queue = queue.Queue()
        self._tts_thread = None
        self._last_activity = time.time()
        self._tts_models = {
            "en": "en_US-hfc_female-medium",
            "pl": "pl_PL-gosia-medium"
        }
        self._tts_instances = {}
        
        # # Download models on initialization
        # for model in self._tts_models.values():
        #     try:
        #         download_piper_model(model)
        #     except Exception as e:
        #         logger.error(f"Error downloading model {model}: {str(e)}")
        #         raise HTTPException(status_code=500, detail=str(e))

    def get_tts(self, language: str):
        if language not in self._tts_instances:
            if language not in self._tts_models:
                raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
            model_path = os.path.join("models", self._tts_models[language])
            if not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Model not found: {model_path}")
            
            # Initialize piper with the model
            config_path = os.path.join(model_path, f"{self._tts_models[language]}.onnx.json")
            model_path = os.path.join(model_path, f"{self._tts_models[language]}.onnx")
            
            if not os.path.exists(config_path) or not os.path.exists(model_path):
                raise HTTPException(status_code=500, detail=f"Model files not found in {model_path}")
            
            self._tts_instances[language] = {
                "config_path": config_path,
                "model_path": model_path
            }
        
        return self._tts_instances[language]

    def stop(self):
        self._stop_event.set()
        if self._tts_thread and self._tts_thread.is_alive():
            try:
                _async_raise(self._tts_thread.ident, SystemExit)
                self._tts_thread.join(timeout=1.0)
            except Exception as e:
                logger.error(f"Error stopping TTS thread: {str(e)}")

    def update_activity(self):
        self._last_activity = time.time()

    def is_client_active(self, timeout=1.0):
        return time.time() - self._last_activity < timeout

    def generate_speech(self, text: str, language: str, output_path: str):
        try:
            self._tts_thread = threading.Thread(
                target=self._run_tts,
                args=(text, language, output_path)
            )
            self._tts_thread.daemon = True
            self._tts_thread.start()
            
            while not self._stop_event.is_set():
                try:
                    result = self._result_queue.get(timeout=0.1)
                    return result
                except queue.Empty:
                    if not self.is_client_active():
                        self._stop_event.set()
                        raise Exception("Client inactive")
                    continue
            
            raise Exception("TTS generation cancelled")
            
        except Exception as e:
            self._result_queue.put(e)
            raise

    def _run_tts(self, text: str, language: str, output_path: str):
        try:
            if not self._stop_event.is_set():
                logger.info(f"Starting TTS generation for language: {language}")
                tts_config = self.get_tts(language)
                logger.info("TTS configuration loaded successfully")
                
                logger.info("Starting text synthesis...")
                # Run piper-tts command
                cmd = [
                    "piper",
                    "--model", tts_config["model_path"],
                    "--config", tts_config["config_path"],
                    "--output_file", output_path
                ]
                
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    input=text,
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                
                if result.returncode != 0:
                    logger.error(f"Piper TTS failed: {result.stderr}")
                    raise Exception(f"Piper TTS failed: {result.stderr}")
                
                logger.info("Text synthesis completed successfully")
                
                logger.info("Converting WAV to MP3...")
                audio_segment = AudioSegment.from_wav(output_path)
                mp3_path = output_path.replace('.wav', '.mp3')
                audio_segment.export(mp3_path, format='mp3', bitrate='192k')
                logger.info(f"MP3 file saved successfully: {mp3_path}")
                
                logger.info("Removing temporary WAV file...")
                os.remove(output_path)
                logger.info("Temporary WAV file removed")
                
                if not self._stop_event.is_set():
                    self._result_queue.put(True)
                    logger.info("TTS generation completed successfully")
        except Exception as e:
            logger.error(f"Error in _run_tts: {str(e)}", exc_info=True)
            if not self._stop_event.is_set():
                self._result_queue.put(e)

def ensure_tmp_directory():
    """Ensure tmp directory exists and has proper permissions."""
    tmp_dir = '/app/tmp'  # Use absolute path in Docker container
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
        os.chmod(tmp_dir, 0o777)
    return tmp_dir

def sanitize_filename(title: str) -> str:
    """Sanitize the title to create a valid filename."""
    if not title:
        return f"speech_{uuid.uuid4().hex[:8]}"
    # Convert special characters to ASCII equivalents
    title = unidecode(title)
    # Remove invalid characters and replace spaces with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '', title)
    sanitized = sanitized.replace(' ', '_')
    # Keep only ASCII characters and basic punctuation
    sanitized = ''.join(char for char in sanitized if ord(char) < 128 or char in '_-.,')
    # Add a UUID to ensure uniqueness
    return f"{sanitized}_{uuid.uuid4().hex[:8]}"

@app.post("/generate")
async def generate_speech(request: TTSRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    logger.info(f"Received TTS request for text: {request.text[:50]}...")
    logger.info(f"Language: {request.language}")
    logger.info(f"Title: {request.title}")
    
    output_path = None
    tts_manager = None
    
    try:
        # Generate filename based on title
        filename = f"{sanitize_filename(request.title)}.wav"
        output_path = os.path.join(ensure_tmp_directory(), filename)
        
        # Create TTS manager
        tts_manager = TTSManager()
        
        # Create a task for TTS generation
        tts_task = asyncio.create_task(
            asyncio.to_thread(
                tts_manager.generate_speech,
                request.text,
                request.language,
                output_path
            )
        )
        
        # Create a task to check for client disconnection
        async def check_disconnection():
            while not tts_task.done():
                if await fastapi_request.is_disconnected():
                    tts_manager.stop()
                    raise Exception("Request cancelled by client")
                tts_manager.update_activity()
                await asyncio.sleep(0.1)
        
        # Run both tasks concurrently
        await asyncio.gather(
            tts_task,
            check_disconnection(),
            return_exceptions=True
        )
        
        # Get the result
        await tts_task
        
        # Update output path to MP3
        mp3_path = output_path.replace('.wav', '.mp3')
        
        # Verify the file exists
        if not os.path.exists(mp3_path):
            logger.error(f"Generated audio file not found at {mp3_path}")
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        # Create a function to stream the file
        async def file_stream():
            try:
                with open(mp3_path, 'rb') as file:
                    while chunk := file.read(8192):
                        yield chunk
            finally:
                # Clean up the file after streaming
                try:
                    os.remove(mp3_path)
                except OSError as e:
                    logger.error(f"Error cleaning up file: {str(e)}")
        
        # Return the audio file
        return StreamingResponse(
            file_stream(),
            media_type='audio/mpeg',
            headers={
                'Content-Disposition': f'attachment; filename="{os.path.basename(mp3_path)}"'
            }
        )
        
    except Exception as e:
        logger.error(f"Error during TTS generation: {str(e)}", exc_info=True)
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                logger.error(f"Error cleaning up file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tts_manager:
            tts_manager.stop()

@app.post("/generate1")
async def generate_speech_piper(request: TTSRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    logger.info(f"Received Piper TTS request for text: {request.text[:50]}...")
    logger.info(f"Language: {request.language}")
    logger.info(f"Title: {request.title}")
    
    output_path = None
    tts_manager = None
    
    try:
        # Generate filename based on title
        filename = f"{sanitize_filename(request.title)}.wav"
        output_path = os.path.join(ensure_tmp_directory(), filename)
        
        # Create TTS manager
        tts_manager = PiperTTSManager()
        
        # Create a task for TTS generation
        tts_task = asyncio.create_task(
            asyncio.to_thread(
                tts_manager.generate_speech,
                request.text,
                request.language,
                output_path
            )
        )
        
        # Create a task to check for client disconnection
        async def check_disconnection():
            while not tts_task.done():
                if await fastapi_request.is_disconnected():
                    tts_manager.stop()
                    raise Exception("Request cancelled by client")
                tts_manager.update_activity()
                await asyncio.sleep(0.1)
        
        # Run both tasks concurrently
        await asyncio.gather(
            tts_task,
            check_disconnection(),
            return_exceptions=True
        )
        
        # Get the result
        await tts_task
        
        # Update output path to MP3
        mp3_path = output_path.replace('.wav', '.mp3')
        
        # Verify the file exists
        if not os.path.exists(mp3_path):
            logger.error(f"Generated audio file not found at {mp3_path}")
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
        # Create a function to stream the file
        async def file_stream():
            try:
                with open(mp3_path, 'rb') as file:
                    while chunk := file.read(8192):
                        yield chunk
            finally:
                # Clean up the file after streaming
                try:
                    os.remove(mp3_path)
                except OSError as e:
                    logger.error(f"Error cleaning up file: {str(e)}")
        
        # Return the audio file
        return StreamingResponse(
            file_stream(),
            media_type='audio/mpeg',
            headers={
                'Content-Disposition': f'attachment; filename="{os.path.basename(mp3_path)}"'
            }
        )
        
    except Exception as e:
        logger.error(f"Error during TTS generation: {str(e)}", exc_info=True)
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError as e:
                logger.error(f"Error cleaning up file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tts_manager:
            tts_manager.stop()