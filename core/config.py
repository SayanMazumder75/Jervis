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
    "openai_model":       "gpt-3.5-turbo",
    "openai_max_tokens":  180,
    "debug_mode":         False,
    "notes_folder":       str(Path.home() / "Desktop" / "JARVIS_Notes"),
    "screenshots_folder": str(Path.home() / "Desktop"),
    "app_paths": {
        "chrome":     r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "firefox":    r"C:\Program Files\Mozilla Firefox\firefox.exe",
        "vscode":     r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
        "notepad":    "notepad.exe",
        "explorer":   "explorer.exe",
        "calculator": "calc.exe",
        "taskmgr":    "taskmgr.exe",
        "paint":      "mspaint.exe",
        "spotify":    r"C:\Users\{user}\AppData\Roaming\Spotify\Spotify.exe",
        "word":       r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        "excel":      r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        "discord":    r"C:\Users\{user}\AppData\Local\Discord\Update.exe",
        "steam":      r"C:\Program Files (x86)\Steam\Steam.exe",
    }
}


class Config:
    def __init__(self):
        data = {k: v for k, v in DEFAULTS.items()}
        if CONFIG_FILE.exists():
            try:
                user_cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                for k, v in user_cfg.items():
                    if k != "app_paths":
                        data[k] = v
                data["app_paths"] = {**DEFAULTS["app_paths"],
                                     **user_cfg.get("app_paths", {})}
            except Exception:
                pass
        else:
            CONFIG_FILE.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")

        for k, v in data.items():
            setattr(self, k, v)

        username = os.environ.get("USERNAME", "User")
        self.app_paths = {k: v.replace("{user}", username)
                          for k, v in self.app_paths.items()}

        Path(self.notes_folder).mkdir(parents=True, exist_ok=True)
        Path(self.screenshots_folder).mkdir(parents=True, exist_ok=True)
