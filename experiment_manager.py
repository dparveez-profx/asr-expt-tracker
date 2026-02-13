#!/usr/bin/env python3
"""
Console-based experiment manager for adversarial attacks on ASR systems.
Uses Textual for TUI and SQLite for data persistence.
"""

import sqlite3
import os
import wave
import json
import numpy as np
from datetime import datetime
from scipy.io import wavfile
from vosk import Model, KaldiRecognizer
import contextlib
import io
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Input, Select, Static, DataTable, Label, TextArea
from textual.screen import Screen
from textual import on
from typing import List, Optional, Tuple

DB_FILE = "experiments.db"
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


class Experiment:
    def __init__(self, id: int = None, name: str = "", description: str = "", 
                 attack_type: str = "", model: str = MODEL_NAME, created_at: str = None):
        self.id = id
        self.name = name
        self.description = description
        self.attack_type = attack_type
        self.model = model
        self.created_at = created_at or datetime.now().isoformat()


class Audio:
    def __init__(self, id: int = None, name: str = "", description: str = "", 
                 file_path: str = "", model: str = MODEL_NAME, transcription: str = "", 
                 experiment_id: Optional[int] = None, created_at: str = None):
        self.id = id
        self.name = name
        self.description = description
        self.file_path = file_path
        self.model = model
        self.transcription = transcription
        self.experiment_id = experiment_id
        self.created_at = created_at or datetime.now().isoformat()

class Database:
    def __init__(self, db_file: str = DB_FILE):
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self):
        with self.conn:
            # Experiments table (existing)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    attack_type TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            # Audios table for new audio management
            # (with optional link to experiment)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audios (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    file_path TEXT NOT NULL,
                    model TEXT NOT NULL,
                    transcription TEXT,
                    experiment_id INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            # Add experiment_id column to existing audios table if missing (for upgrade)
            try:
                self.conn.execute("ALTER TABLE audios ADD COLUMN experiment_id INTEGER")
            except sqlite3.OperationalError:
                pass  # column already exists

    def create_experiment(self, exp: Experiment) -> int:
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO experiments (name, description, attack_type, model, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (exp.name, exp.description, exp.attack_type, exp.model, exp.created_at))
            return cursor.lastrowid

    def get_all_experiments(self) -> List[Experiment]:
        cursor = self.conn.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        return [Experiment(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            attack_type=row["attack_type"],
            model=row["model"],
            created_at=row["created_at"]
        ) for row in cursor.fetchall()]

    def get_experiment(self, exp_id: int) -> Optional[Experiment]:
        cursor = self.conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        if row:
            return Experiment(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                attack_type=row["attack_type"],
                model=row["model"],
                created_at=row["created_at"]
            )
        return None

    def update_experiment(self, exp: Experiment) -> bool:
        with self.conn:
            cursor = self.conn.execute("""
                UPDATE experiments 
                SET name = ?, description = ?, attack_type = ?, model = ?
                WHERE id = ?
            """, (exp.name, exp.description, exp.attack_type, exp.model, exp.id))
            return cursor.rowcount > 0

    def delete_experiment(self, exp_id: int) -> bool:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            return cursor.rowcount > 0

    # Audio CRUD methods
    def create_audio(self, audio: Audio) -> int:
        with self.conn:
            cursor = self.conn.execute("""
                INSERT INTO audios (name, description, file_path, model, transcription, experiment_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (audio.name, audio.description, audio.file_path, audio.model, audio.transcription, audio.experiment_id, audio.created_at))
            return cursor.lastrowid

    def get_all_audios(self) -> List[Audio]:
        cursor = self.conn.execute("SELECT * FROM audios ORDER BY created_at DESC")
        return [Audio(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            file_path=row["file_path"],
            model=row["model"],
            transcription=row["transcription"] or "",
            experiment_id=row["experiment_id"],
            created_at=row["created_at"]
        ) for row in cursor.fetchall()]

    def get_audio(self, audio_id: int) -> Optional[Audio]:
        cursor = self.conn.execute("SELECT * FROM audios WHERE id = ?", (audio_id,))
        row = cursor.fetchone()
        if row:
            return Audio(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                file_path=row["file_path"],
                model=row["model"],
                transcription=row["transcription"] or "",
                experiment_id=row["experiment_id"],
                created_at=row["created_at"]
            )
        return None

    def update_audio(self, audio: Audio) -> bool:
        with self.conn:
            cursor = self.conn.execute("""
                UPDATE audios 
                SET name = ?, description = ?, file_path = ?, model = ?, transcription = ?, experiment_id = ?
                WHERE id = ?
            """, (audio.name, audio.description, audio.file_path, audio.model, audio.transcription, audio.experiment_id, audio.id))
            return cursor.rowcount > 0

    def delete_audio(self, audio_id: int) -> bool:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM audios WHERE id = ?", (audio_id,))
            return cursor.rowcount > 0

    # NUKE: clear all data (for convenience/reset)
    def nuke_all_data(self) -> bool:
        try:
            with self.conn:
                self.conn.execute("DELETE FROM experiments")
                self.conn.execute("DELETE FROM audios")
            return True
        except Exception:
            return False

    def close(self):
        self.conn.close()

class CreateExperimentScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Create New Experiment", classes="title")
            yield Input(placeholder="Experiment Name", id="name_input")
            yield TextArea(placeholder="Description (optional)", id="desc_input", classes="description")
            yield Select(
                [("CW", "CW"), ("PGD", "PGD"), ("AdvReverb", "AdvReverb")],
                id="attack_select",
                value="CW"  # default to prevent NoSelection error
            )
            # Model select (uniform with audios; dynamic from models/)
            models = get_available_models()
            model_options = [(m, m) for m in models]
            yield Select(
                model_options,
                id="model_select",
                value=MODEL_NAME
            )
            with Horizontal():
                yield Button("Create", id="create_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn", variant="error")

    @on(Button.Pressed, "#create_btn")
    def create_experiment(self):
        name = self.query_one("#name_input", Input).value.strip()
        if not name:
            self.notify("Experiment name is required!", severity="error")
            return
        
        desc = self.query_one("#desc_input", TextArea).text.strip()
        attack_type = self.query_one("#attack_select", Select).value
        # Guard against NoSelection or invalid type (Textual can return NoSelection object if not defaulted)
        if not isinstance(attack_type, str) or attack_type not in ["CW", "PGD", "AdvReverb"]:
            self.notify("Please select a valid attack type (CW, PGD, or AdvReverb)!", severity="error")
            return
        model = self.query_one("#model_select", Select).value
        
        db = Database()
        exp = Experiment(name=name, description=desc, attack_type=attack_type, model=model)
        exp_id = db.create_experiment(exp)
        db.close()
        
        self.notify(f"Experiment created with ID: {exp_id}")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())

class ViewExperimentsScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Experiments", classes="title")
            with Horizontal():
                yield Button("Refresh", id="refresh_btn")
                yield Button("Back", id="back_btn")
            yield ScrollableContainer(DataTable(id="exp_table"), id="table_container")

    def on_mount(self):
        # Use call_after_refresh to ensure widgets (like DataTable) are fully mounted
        # before querying, avoiding NoMatches error
        self.call_after_refresh(self.refresh_table)

    def refresh_table(self):
        try:
            table = self.query_one("#exp_table", DataTable)
        except Exception:  # Fallback for any mount timing issues
            # If still not ready, schedule again
            self.call_after_refresh(self.refresh_table)
            return
        # Clear both rows AND columns to prevent duplicates on repeated refreshes (e.g. on_mount + button)
        # (DataTable.add_columns appends; clear(columns=True) resets headers)
        table.clear(columns=True)
        table.add_columns("ID", "Name", "Attack Type", "Model", "Associated Audios", "Created At", "Description")
        
        db = Database()
        exps = db.get_all_experiments()
        # Get audio counts/IDs per exp for display
        audios = db.get_all_audios()
        db.close()
        audio_map = {}
        for a in audios:
            if a.experiment_id:
                if a.experiment_id not in audio_map:
                    audio_map[a.experiment_id] = []
                audio_map[a.experiment_id].append(str(a.id))
        
        for exp in exps:
            desc_short = (exp.description[:50] + "...") if len(exp.description) > 50 else exp.description
            audio_ids = audio_map.get(exp.id, [])
            audio_info = f"{len(audio_ids)}: {', '.join(audio_ids)}" if audio_ids else "None"
            table.add_row(
                str(exp.id),
                exp.name,
                exp.attack_type,
                exp.model,
                audio_info,
                exp.created_at,
                desc_short
            )

    @on(Button.Pressed, "#refresh_btn")
    def on_refresh_pressed(self):
        # Renamed to avoid conflict with Textual's inherited refresh() method (which expects layout args)
        self.refresh_table()

    @on(Button.Pressed, "#back_btn")
    def back(self):
        self.app.switch_screen(MainScreen())

class UpdateExperimentScreen(Screen):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, exp_id: int):
        super().__init__()
        self.exp_id = exp_id
        self.exp = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Update Experiment", classes="title")
            yield Input(placeholder="Experiment Name", id="name_input")
            yield TextArea(placeholder="Description (optional)", id="desc_input", classes="description")
            yield Select(
                [("CW", "CW"), ("PGD", "PGD"), ("AdvReverb", "AdvReverb")],
                id="attack_select",
                value="CW"  # default to prevent NoSelection error
            )
            # Model select (uniform with audios; dynamic from models/)
            models = get_available_models()
            model_options = [(m, m) for m in models]
            yield Select(
                model_options,
                id="model_select",
                value=MODEL_NAME
            )
            with Horizontal():
                yield Button("Update", id="update_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn", variant="error")

    def on_mount(self):
        db = Database()
        self.exp = db.get_experiment(self.exp_id)
        db.close()
        
        if self.exp:
            self.query_one("#name_input", Input).value = self.exp.name
            self.query_one("#desc_input", TextArea).text = self.exp.description
            select = self.query_one("#attack_select", Select)
            select.value = self.exp.attack_type
            # Set model
            model_select = self.query_one("#model_select", Select)
            model_select.value = self.exp.model
        else:
            self.notify("Experiment not found!", severity="error")
            self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#update_btn")
    def update_experiment(self):
        if not self.exp:
            return
        name = self.query_one("#name_input", Input).value.strip()
        if not name:
            self.notify("Experiment name is required!", severity="error")
            return
        
        desc = self.query_one("#desc_input", TextArea).text.strip()
        attack_type = self.query_one("#attack_select", Select).value
        # Guard against NoSelection or invalid type (Textual can return NoSelection object if not defaulted)
        if not isinstance(attack_type, str) or attack_type not in ["CW", "PGD", "AdvReverb"]:
            self.notify("Please select a valid attack type (CW, PGD, or AdvReverb)!", severity="error")
            return
        model = self.query_one("#model_select", Select).value
        
        db = Database()
        self.exp.name = name
        self.exp.description = desc
        self.exp.attack_type = attack_type
        self.exp.model = model
        success = db.update_experiment(self.exp)
        db.close()
        
        if success:
            self.notify("Experiment updated successfully!")
        else:
            self.notify("Update failed!", severity="error")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())

    def action_cancel(self):
        self.app.switch_screen(MainScreen())

class DeleteExperimentScreen(Screen):
    def __init__(self, exp_id: int):
        super().__init__()
        self.exp_id = exp_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Delete Experiment", classes="title")
            yield Label("Are you sure you want to delete this experiment?", id="confirm_label")
            with Horizontal():
                yield Button("Yes, Delete", id="delete_btn", variant="error")
                yield Button("Cancel", id="cancel_btn", variant="primary")

    def on_mount(self):
        db = Database()
        exp = db.get_experiment(self.exp_id)
        db.close()
        if exp:
            label = self.query_one("#confirm_label", Label)
            label.update(f"Are you sure you want to delete experiment '{exp.name}' (ID: {exp.id})?")
        else:
            self.notify("Experiment not found!", severity="error")
            self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#delete_btn")
    def delete_experiment(self):
        db = Database()
        success = db.delete_experiment(self.exp_id)
        db.close()
        
        if success:
            self.notify("Experiment deleted successfully!")
        else:
            self.notify("Delete failed!", severity="error")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())

class MainScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("ASR Adversarial Experiments Manager", classes="title")
            # Side-by-side sections for uniformity (Experiments left, Audios right)
            with Horizontal():
                # Experiments section
                with Vertical(classes="section-column"):
                    yield Label("Experiments:", classes="section-title")
                    yield Button("Create New Experiment", id="create_exp_btn", variant="primary")
                    yield Button("View Experiments", id="view_exp_btn")
                    yield Button("Update Experiment", id="update_exp_btn")
                    yield Button("Delete Experiment", id="delete_exp_btn")
                # Audios section
                with Vertical(classes="section-column"):
                    yield Label("Audios:", classes="section-title")
                    yield Button("Add Audio", id="add_audio_btn", variant="primary")
                    yield Button("View Audios", id="view_audio_btn")
                    yield Button("Update Audio", id="update_audio_btn")
                    yield Button("Delete Audio", id="delete_audio_btn")
            # NUKE and Exit at bottom
            with Vertical():
                yield Button("NUKE ALL DATA", id="nuke_btn", variant="error")
                yield Button("Exit", id="exit_btn", variant="error")

    @on(Button.Pressed, "#create_exp_btn")
    def create_exp(self):
        self.app.push_screen(CreateExperimentScreen())

    @on(Button.Pressed, "#view_exp_btn")
    def view_exp(self):
        self.app.push_screen(ViewExperimentsScreen())

    @on(Button.Pressed, "#update_exp_btn")
    def update_exp(self):
        # For simplicity, ask for ID via input
        self.app.push_screen(UpdateExperimentPrompt())

    @on(Button.Pressed, "#delete_exp_btn")
    def delete_exp(self):
        self.app.push_screen(DeleteExperimentPrompt())

    # Audio handlers (new)
    @on(Button.Pressed, "#add_audio_btn")
    def add_audio(self):
        self.app.push_screen(CreateAudioScreen())

    @on(Button.Pressed, "#view_audio_btn")
    def view_audio(self):
        self.app.push_screen(ViewAudiosScreen())

    @on(Button.Pressed, "#update_audio_btn")
    def update_audio(self):
        self.app.push_screen(UpdateAudioPrompt())

    @on(Button.Pressed, "#delete_audio_btn")
    def delete_audio(self):
        self.app.push_screen(DeleteAudioPrompt())

    # NUKE handler
    @on(Button.Pressed, "#nuke_btn")
    def nuke(self):
        self.app.push_screen(NukeConfirmScreen())

    @on(Button.Pressed, "#exit_btn")
    def exit_app(self):
        self.app.exit()

class UpdateExperimentPrompt(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Enter Experiment ID to Update:")
            yield Input(placeholder="Experiment ID", id="id_input")
            with Horizontal():
                yield Button("Update", id="proceed_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn")

    @on(Button.Pressed, "#proceed_btn")
    def proceed(self):
        try:
            exp_id = int(self.query_one("#id_input", Input).value.strip())
            self.app.push_screen(UpdateExperimentScreen(exp_id))
        except ValueError:
            self.notify("Invalid ID! Must be integer.", severity="error")

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())

class DeleteExperimentPrompt(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Enter Experiment ID to Delete:")
            yield Input(placeholder="Experiment ID", id="id_input")
            with Horizontal():
                yield Button("Delete", id="proceed_btn", variant="error")
                yield Button("Cancel", id="cancel_btn")

    @on(Button.Pressed, "#proceed_btn")
    def proceed(self):
        try:
            exp_id = int(self.query_one("#id_input", Input).value.strip())
            self.app.push_screen(DeleteExperimentScreen(exp_id))
        except ValueError:
            self.notify("Invalid ID! Must be integer.", severity="error")

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


# Audio management screens (new phase)
class CreateAudioScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Add New Audio", classes="title")
            yield Input(placeholder="Audio Name", id="name_input")
            yield TextArea(placeholder="Description (optional)", id="desc_input", classes="description")
            yield Input(placeholder="Audio File Path (e.g. /path/to/audio.wav)", id="path_input")
            yield Input(placeholder="Experiment ID (optional, for mapping)", id="exp_id_input")
            yield Select(
                # Dynamically could scan models/, but for now fixed to available Vosk model
                [(MODEL_NAME, MODEL_NAME)],
                id="model_select",
                value=MODEL_NAME
            )
            # Non-intrusive list of available experiments
            yield Label("Available Experiments (ID: Name - Desc):", classes="section-title")
            yield Static(id="exp_list_static", classes="exp-list")
            with Horizontal():
                yield Button("Add & Transcribe", id="add_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn", variant="error")

    def on_mount(self):
        # Populate non-intrusive list of experiments for mapping reference
        self.populate_exp_list()

    def populate_exp_list(self):
        db = Database()
        exps = db.get_all_experiments()
        db.close()
        summary = "\n".join(
            [f"{e.id}: {e.name} - {e.description[:30]}..." for e in exps]
        ) or "No experiments yet."
        self.query_one("#exp_list_static", Static).update(summary)

    @on(Button.Pressed, "#add_btn")
    def add_audio(self):
        name = self.query_one("#name_input", Input).value.strip()
        if not name:
            self.notify("Audio name is required!", severity="error")
            return
        file_path = self.query_one("#path_input", Input).value.strip()
        if not file_path or not os.path.exists(file_path):
            self.notify("Valid audio file path is required!", severity="error")
            return
        desc = self.query_one("#desc_input", TextArea).text.strip()
        model = self.query_one("#model_select", Select).value
        # Optional exp mapping
        exp_id_str = self.query_one("#exp_id_input", Input).value.strip()
        experiment_id = int(exp_id_str) if exp_id_str.isdigit() else None
        if experiment_id is not None:
            # Validate exp exists
            db_check = Database()
            if not db_check.get_experiment(experiment_id):
                db_check.close()
                self.notify(f"Experiment ID {experiment_id} does not exist!", severity="error")
                return
            db_check.close()
        
        # Perform transcription
        transcription = transcribe_audio(file_path, model)
        self.notify("Transcription completed (see result in audio details).")
        
        db = Database()
        audio = Audio(name=name, description=desc, file_path=file_path, model=model, 
                      transcription=transcription, experiment_id=experiment_id)
        audio_id = db.create_audio(audio)
        db.close()
        
        self.notify(f"Audio added with ID: {audio_id} and transcription saved.")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


class ViewAudiosScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Audios", classes="title")
            with Horizontal():
                yield Button("Refresh", id="refresh_btn")
                yield Button("Back", id="back_btn")
            yield ScrollableContainer(DataTable(id="audio_table"), id="table_container")

    def on_mount(self):
        self.call_after_refresh(self.refresh_table)

    def refresh_table(self):
        try:
            table = self.query_one("#audio_table", DataTable)
        except Exception:
            self.call_after_refresh(self.refresh_table)
            return
        table.clear(columns=True)
        table.add_columns("ID", "Name", "Model", "Exp ID", "Transcription Preview", "Created At", "Path")
        
        db = Database()
        audios = db.get_all_audios()
        db.close()
        
        for audio in audios:
            trans_preview = (audio.transcription[:50] + "...") if len(audio.transcription) > 50 else audio.transcription
            exp_id_str = str(audio.experiment_id) if audio.experiment_id else "None"
            table.add_row(
                str(audio.id),
                audio.name,
                audio.model,
                exp_id_str,
                trans_preview,
                audio.created_at,
                audio.file_path
            )

    @on(Button.Pressed, "#refresh_btn")
    def on_refresh_pressed(self):
        self.refresh_table()

    @on(Button.Pressed, "#back_btn")
    def back(self):
        self.app.switch_screen(MainScreen())


class UpdateAudioPrompt(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Enter Audio ID to Update:")
            yield Input(placeholder="Audio ID", id="id_input")
            with Horizontal():
                yield Button("Update", id="proceed_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn")

    @on(Button.Pressed, "#proceed_btn")
    def proceed(self):
        try:
            audio_id = int(self.query_one("#id_input", Input).value.strip())
            self.app.push_screen(UpdateAudioScreen(audio_id))
        except ValueError:
            self.notify("Invalid ID! Must be integer.", severity="error")

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


class UpdateAudioScreen(Screen):
    def __init__(self, audio_id: int):
        super().__init__()
        self.audio_id = audio_id
        self.audio = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Update Audio", classes="title")
            yield Input(placeholder="Audio Name", id="name_input")
            yield TextArea(placeholder="Description (optional)", id="desc_input", classes="description")
            yield Input(placeholder="Audio File Path", id="path_input")
            yield Input(placeholder="Experiment ID (optional, for mapping)", id="exp_id_input")
            yield Select(
                [(MODEL_NAME, MODEL_NAME)],
                id="model_select",
                value=MODEL_NAME
            )
            # Non-intrusive list of available experiments for mapping reference
            yield Label("Available Experiments (ID: Name - Desc):", classes="section-title")
            yield Static(id="exp_list_static", classes="exp-list")
            with Horizontal():
                yield Button("Update", id="update_btn", variant="primary")
                yield Button("Cancel", id="cancel_btn", variant="error")

    def on_mount(self):
        db = Database()
        self.audio = db.get_audio(self.audio_id)
        db.close()
        if self.audio:
            self.query_one("#name_input", Input).value = self.audio.name
            self.query_one("#desc_input", TextArea).text = self.audio.description
            self.query_one("#path_input", Input).value = self.audio.file_path
            # Set exp ID if linked
            if self.audio.experiment_id:
                self.query_one("#exp_id_input", Input).value = str(self.audio.experiment_id)
            # Note: transcription not editable here; re-transcribe if path changes on save
            # Populate exp list
            self.populate_exp_list()
        else:
            self.notify("Audio not found!", severity="error")
            self.app.switch_screen(MainScreen())

    def populate_exp_list(self):
        # Reuse helper logic
        db = Database()
        exps = db.get_all_experiments()
        db.close()
        summary = "\n".join(
            [f"{e.id}: {e.name} - {e.description[:30]}..." for e in exps]
        ) or "No experiments yet."
        self.query_one("#exp_list_static", Static).update(summary)

    @on(Button.Pressed, "#update_btn")
    def update_audio(self):
        if not self.audio:
            return
        name = self.query_one("#name_input", Input).value.strip()
        if not name:
            self.notify("Audio name is required!", severity="error")
            return
        file_path = self.query_one("#path_input", Input).value.strip()
        if not file_path:
            self.notify("File path is required!", severity="error")
            return
        desc = self.query_one("#desc_input", TextArea).text.strip()
        model = self.query_one("#model_select", Select).value
        # Optional exp mapping
        exp_id_str = self.query_one("#exp_id_input", Input).value.strip()
        experiment_id = int(exp_id_str) if exp_id_str.isdigit() else None
        if experiment_id is not None:
            # Validate exp exists
            db_check = Database()
            if not db_check.get_experiment(experiment_id):
                db_check.close()
                self.notify(f"Experiment ID {experiment_id} does not exist!", severity="error")
                return
            db_check.close()
        
        # Re-transcribe if path changed (simple always re-do)
        transcription = transcribe_audio(file_path, model)
        
        db = Database()
        self.audio.name = name
        self.audio.description = desc
        self.audio.file_path = file_path
        self.audio.model = model
        self.audio.transcription = transcription
        self.audio.experiment_id = experiment_id
        success = db.update_audio(self.audio)
        db.close()
        
        if success:
            self.notify("Audio updated with new transcription!")
        else:
            self.notify("Update failed!", severity="error")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


class DeleteAudioPrompt(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Enter Audio ID to Delete:")
            yield Input(placeholder="Audio ID", id="id_input")
            with Horizontal():
                yield Button("Delete", id="proceed_btn", variant="error")
                yield Button("Cancel", id="cancel_btn")

    @on(Button.Pressed, "#proceed_btn")
    def proceed(self):
        try:
            audio_id = int(self.query_one("#id_input", Input).value.strip())
            self.app.push_screen(DeleteAudioScreen(audio_id))
        except ValueError:
            self.notify("Invalid ID! Must be integer.", severity="error")

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


class DeleteAudioScreen(Screen):
    def __init__(self, audio_id: int):
        super().__init__()
        self.audio_id = audio_id

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("Delete Audio", classes="title")
            yield Label("Are you sure?", id="confirm_label")
            with Horizontal():
                yield Button("Yes, Delete", id="delete_btn", variant="error")
                yield Button("Cancel", id="cancel_btn", variant="primary")

    def on_mount(self):
        db = Database()
        audio = db.get_audio(self.audio_id)
        db.close()
        if audio:
            label = self.query_one("#confirm_label", Label)
            label.update(f"Are you sure you want to delete audio '{audio.name}' (ID: {audio.id})?")
        else:
            self.notify("Audio not found!", severity="error")
            self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#delete_btn")
    def delete_audio(self):
        db = Database()
        success = db.delete_audio(self.audio_id)
        db.close()
        if success:
            self.notify("Audio deleted successfully!")
        else:
            self.notify("Delete failed!", severity="error")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


# NUKE confirm screen (clear all data)
class NukeConfirmScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        with Container(id="main"):
            yield Label("NUKE ALL DATA", classes="title")
            yield Label("This will permanently delete ALL experiments and audios from the database. Are you sure?", id="confirm_label")
            with Horizontal():
                yield Button("YES, NUKE IT", id="nuke_btn", variant="error")
                yield Button("Cancel", id="cancel_btn", variant="primary")

    @on(Button.Pressed, "#nuke_btn")
    def nuke_data(self):
        db = Database()
        success = db.nuke_all_data()
        db.close()
        if success:
            self.notify("Database nuked! All data cleared.", severity="warning")
        else:
            self.notify("Nuke failed!", severity="error")
        self.app.switch_screen(MainScreen())

    @on(Button.Pressed, "#cancel_btn")
    def cancel(self):
        self.app.switch_screen(MainScreen())


class ExperimentManagerApp(App):
    """Main Textual application for experiment management."""
    
    CSS = """
    Screen {
        align: center middle;
    }
    #main {
        width: 80%;
        height: 80%;
        border: solid $primary;
        padding: 2;
        overflow-y: auto;
    }
    .title {
        text-align: center;
        width: 100%;
        margin-bottom: 2;
        text-style: bold;
        height: auto;
    }
    .section-title {
        text-align: left;
        width: 100%;
        margin: 1 0;
        text-style: bold;
        color: $accent;
    }
    .exp-list {
        width: 100%;
        height: 8;
        border: solid $secondary;
        padding: 1;
        overflow-y: auto;
        text-style: italic;
    }
    Input, TextArea, Select {
        margin-bottom: 1;
        width: 100%;
        height: auto;
    }
    .description {
        height: 5;
    }
    DataTable {
        margin-top: 1;
        height: 100%;
        width: 100%;
    }
    #table_container {
        height: 70%;
        width: 100%;
        border: solid $secondary;
    }
    Button {
        margin: 1;
        width: 40%;
        height: auto;
    }
    Horizontal {
        align: center middle;
        height: auto;
        width: 100%;
    }
    Vertical {
        align: center middle;
        height: auto;
        width: 100%;
    }
    .section-column {
        width: 48%;
        margin-right: 2;
    }
    """

    def compose(self) -> ComposeResult:
        # Base compose; initial screen pushed in on_mount for reliable rendering
        yield Container(id="root")

    def on_mount(self):
        # Initialize DB
        db = Database()
        db.close()
        # Push main screen explicitly to ensure proper initial layout/rendering
        self.push_screen(MainScreen())

if __name__ == "__main__":
    app = ExperimentManagerApp()
    app.run()
