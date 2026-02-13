# zed-base

## ASR Adversarial Experiments Manager

A console-based application for managing experiments with adversarial attacks on automatic speech recognition (ASR) systems.

Built with:
- Python
- Textual (for TUI)
- SQLite (for experiment storage)

### Features
- **Experiments CRUD**: Create/view/update/delete with name, optional desc, attack type (CW/PGD/AdvReverb), and selectable model (from `models/` dir).
- **Audios CRUD**: Create/view/update/delete with name, optional desc, local WAV path, selectable model (from `models/` dir), auto-transcription via Vosk (resamples formats for compatibility).
  - Optional mapping to experiment ID (validated; list shown on screens).
- **View enhancements**: Audios show linked exp ID; Experiments show associated audio count/IDs.
- **NUKE option**: Clear all DB data (with confirm).
- **Vosk integration**: Transcription on audio save/update (logs to `transcription.log`; console suppressed).
- Models: Dynamic from `models/` (default `vosk-model-small-en-us-0.15`).

### Usage
Run the application:
```bash
python experiment_manager.py
```

Main menu (side-by-side sections):
- Experiments: CRUD + model select.
- Audios: Add (with transcription + exp link), View (exp ID), Update/Delete.
- NUKE ALL DATA / Exit at bottom.

Use dummy_test_audio.wav for quick tests (valid WAV; transcription may be blank for non-speech).

### Database
- `experiments.db` (SQLite): `experiments` and `audios` tables (with optional exp-audio links).

### Notes
- Audio: WAV preferred (16kHz mono best); resamples others.
- Logs: Transcription details/errors in `transcription.log`.
- Future: Attacks, more audio processing, etc.
