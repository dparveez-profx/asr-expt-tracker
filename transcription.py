import os
import wave
import json
import numpy as np
from datetime import datetime
from scipy.io import wavfile
from vosk import Model, KaldiRecognizer
import contextlib
import io

MODEL_NAME = "vosk-model-small-en-us-0.15"

def get_available_models():
    """Scan models/ dir for subdirectories (Vosk models)."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return [MODEL_NAME]
    return [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

def transcribe_audio(file_path: str, model_name: str = MODEL_NAME) -> str:
    """Transcribe audio file using Vosk model. Handles/resamples various WAV formats (e.g. format 3/float)
    to 16kHz mono PCM using scipy/numpy for compatibility. Returns transcription or error.
    Logs details to transcription.log for debugging."""
    if not os.path.exists(file_path):
        return f"Error: Audio file not found at {file_path}"
    if not file_path.lower().endswith('.wav'):
        return "Error: Only WAV files supported."
    
    log_file = "transcription.log"
    try:
        # Log input details for debug
        with open(log_file, "a") as log:
            log.write(f"[{datetime.now()}] Processing: {file_path}\n")
        
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            return f"Error: Model {model_name} not found in models/ directory."
        
        # Read audio with scipy (handles format 3/float, int, etc.)
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Log audio properties
        with open(log_file, "a") as log:
            log.write(f"  Original: rate={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}\n")
        
        # Convert to mono if stereo (handle float/int)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        # Normalize/convert to int16 for Vosk
        if audio_data.dtype != np.int16:
            # Scale float [-1,1] or int to int16
            if audio_data.dtype in [np.float32, np.float64]:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Resample to 16kHz if needed
        target_rate = 16000
        if sample_rate != target_rate:
            num_samples = int(len(audio_data) * target_rate / sample_rate)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), num_samples),
                np.arange(len(audio_data)),
                audio_data
            ).astype(np.int16)
            sample_rate = target_rate
        
        # Log after processing
        with open(log_file, "a") as log:
            log.write(f"  Processed: rate={sample_rate}, len={len(audio_data)}\n")
        
        # Now use wave/Kaldi with resampled PCM (suppress Vosk/Kaldi stdout/stderr logs)
        # Use devnull for full quiet (C++ logs sometimes bypass StringIO)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                model = Model(model_path)
        # Write temp resampled WAV
        temp_wav = "/tmp/resampled_temp.wav"
        wavfile.write(temp_wav, sample_rate, audio_data)
        
        # Redirect both for KaldiRecognizer
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                with wave.open(temp_wav, "rb") as wf:
                    rec = KaldiRecognizer(model, wf.getframerate())
                    rec.SetWords(True)
                    transcription = ""
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            transcription += result.get("text", "") + " "
                    final = json.loads(rec.FinalResult())
                    transcription += final.get("text", "")
        
        # Cleanup temp
        if os.path.exists(temp_wav):
            os.unlink(temp_wav)
        
        # Return blank on no speech (less misleading for dummy/quiet audio)
        trans = transcription.strip()
        return trans if trans else ""
    except Exception as e:
        # Log full error
        with open(log_file, "a") as log:
            log.write(f"  ERROR: {str(e)}\n")
            import traceback
            log.write(traceback.format_exc() + "\n")
        return f"Transcription error: {str(e)}"
