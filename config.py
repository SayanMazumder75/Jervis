import json, os
from pathlib import Path

BASE_DIR    = Path(__file__).parent.parent.resolve()
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULTS = {
    "wake_word":          "jarvis",
    "master_name":        "Sir",
    "tts_rate":           160,
    "tts_volume":         1.0,
    "tts_voice_index":    0,
    "energy_threshold":   300,
    "pause_threshold":    0.6,
    "listen_timeout":     7,
    "phrase_time_limit":  10,
    "idle_strikes_limit": 3,
    "openai_model":       "gpt-4o-mini",   # fast + cheap + smart
    "debug_mode":         False,
    "notes_folder":       str(Path.home() / "Desktop" / "JARVIS_Notes"),
    "screenshots_folder": str(Path.home() / "Desktop"),
}


class Config:
    def __init__(self):
        data = dict(DEFAULTS)
        if CONFIG_FILE.exists():
            try:
                saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                data.update(saved)
            except Exception:
                pass
        else:
            CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")

        for k, v in data.items():
            setattr(self, k, v)

        Path(self.notes_folder).mkdir(parents=True, exist_ok=True)
        Path(self.screenshots_folder).mkdir(parents=True, exist_ok=True)
