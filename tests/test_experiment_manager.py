import pytest
import os
import tempfile
import sqlite3
from transcription import transcribe_audio, get_available_models, MODEL_NAME
from database import Database, Experiment, Audio

# Test data
DUMMY_AUDIO = "dummy_test_audio.wav"

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    db_file = tempfile.mktemp(suffix=".db")
    db = Database(db_file)
    yield db
    db.close()
    if os.path.exists(db_file):
        os.unlink(db_file)

def test_get_available_models():
    """Test getting available models."""
    models = get_available_models()
    assert isinstance(models, list)
    assert MODEL_NAME in models or len(models) > 0

def test_transcribe_audio_exists():
    """Test transcription with existing dummy audio."""
    result = transcribe_audio(DUMMY_AUDIO)
    assert isinstance(result, str)
    # Dummy may return empty or some text; just check no error
    assert not result.startswith("Error:")

def test_transcribe_audio_nonexistent():
    """Test transcription with non-existent file."""
    result = transcribe_audio("nonexistent.wav")
    assert result.startswith("Error: Audio file not found")

def test_transcribe_audio_non_wav():
    """Test with non-WAV file."""
    # Create a temp non-wav
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        temp_file = f.name
    result = transcribe_audio(temp_file)
    os.unlink(temp_file)
    assert result == "Error: Only WAV files supported."

@pytest.mark.skipif(not os.path.exists("models/" + MODEL_NAME), reason="Model not available")
def test_transcribe_audio_model_error():
    """Test with invalid model."""
    result = transcribe_audio(DUMMY_AUDIO, "invalid_model")
    assert result.startswith("Error: Model")

def test_database_create_experiment(temp_db):
    """Test creating an experiment."""
    exp = Experiment(name="Test Exp", description="Test", attack_type="CW", model=MODEL_NAME)
    exp_id = temp_db.create_experiment(exp)
    assert exp_id > 0
    retrieved = temp_db.get_experiment(exp_id)
    assert retrieved is not None
    assert retrieved.name == "Test Exp"

def test_database_create_audio(temp_db):
    """Test creating an audio."""
    audio = Audio(name="Test Audio", description="Test", file_path=DUMMY_AUDIO, model=MODEL_NAME, transcription="test")
    audio_id = temp_db.create_audio(audio)
    assert audio_id > 0
    retrieved = temp_db.get_audio(audio_id)
    assert retrieved is not None
    assert retrieved.name == "Test Audio"

def test_database_nuke(temp_db):
    """Test nuking data."""
    # Add some data
    exp = Experiment(name="Test", attack_type="CW", model=MODEL_NAME)
    temp_db.create_experiment(exp)
    audio = Audio(name="Test", file_path=DUMMY_AUDIO, model=MODEL_NAME)
    temp_db.create_audio(audio)
    
    assert len(temp_db.get_all_experiments()) > 0
    assert len(temp_db.get_all_audios()) > 0
    
    success = temp_db.nuke_all_data()
    assert success
    assert len(temp_db.get_all_experiments()) == 0
    assert len(temp_db.get_all_audios()) == 0

# Note: More advanced tests for UI would require Textual's testing framework,
# but this covers core functions.
