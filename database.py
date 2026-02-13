import sqlite3
from typing import List, Optional
from datetime import datetime

DB_FILE = "experiments.db"

class Experiment:
    def __init__(self, id: int = None, name: str = "", description: str = "", 
                 attack_type: str = "", model: str = "", created_at: str = None):
        self.id = id
        self.name = name
        self.description = description
        self.attack_type = attack_type
        self.model = model
        self.created_at = created_at or datetime.now().isoformat()

class Audio:
    def __init__(self, id: int = None, name: str = "", description: str = "", 
                 file_path: str = "", model: str = "", transcription: str = "", 
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
