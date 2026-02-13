#!/usr/bin/env python3
"""
Console-based experiment manager for adversarial attacks on ASR systems.
Uses Textual for TUI and SQLite for data persistence.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Button, Input, Select, Static, DataTable, Label, TextArea
from textual.screen import Screen
from textual import on
from typing import List, Optional
import os

# Local modules
from transcription import transcribe_audio, get_available_models, MODEL_NAME
from database import Database, Experiment, Audio



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
    
    CSS_PATH = "styles.css"

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
