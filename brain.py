"""
core/brain.py  —  JARVIS Brain
Orchestrates wake-word loop, voice, and AI executor.
"""
import threading, logging
from typing import Callable, Optional

from core.voice    import TTSEngine, VoiceEngine
from core.executor import AIExecutor

log = logging.getLogger("jarvis")

SLEEP_WORDS = ("go to sleep", "standby", "goodbye", "bye",
               "sleep now", "that will be all", "shut up")


class JarvisBrain:
    def __init__(self, cfg):
        self.cfg  = cfg
        self.M    = cfg.master_name

        # Voice systems
        self.tts   = TTSEngine(rate=cfg.tts_rate,
                               volume=cfg.tts_volume,
                               voice_index=cfg.tts_voice_index)
        self.voice = VoiceEngine(cfg)

        # AI executor (no hardcoded commands!)
        self.executor = AIExecutor(cfg, self._listen_input)

        # Wire status events → HUD
        self.tts.on_start  = lambda: self._set_status("speaking")
        self.tts.on_end    = lambda: self._set_status(
            "listening" if self._awake else "standby")
        self.voice.status_cb = self._set_status

        # HUD callbacks (set by HUD after construction)
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_wake:   Optional[Callable[[], None]]    = None

        self._awake  = False
        self.running = False

    # ── Start ──────────────────────────────────────────────────────── #
    def start(self):
        self.running = True
        threading.Thread(target=self._boot, daemon=True, name="JARVIS").start()

    def _boot(self):
        self._set_status("calibrating")
        self.voice.calibrate()

        if self.executor.ai_available:
            self.tts.speak_sync(
                f"JARVIS online. Good day, {self.M}. "
                "AI systems fully operational. Say my name to activate.")
        else:
            self.tts.speak_sync(
                f"JARVIS online. Good day, {self.M}. "
                "Note: AI key not detected. Running in basic mode. "
                "Say my name to activate.")
        self._wake_loop()

    # ── Wake-word standby loop ─────────────────────────────────────── #
    def _wake_loop(self):
        log.info(f"Waiting for wake word: '{self.cfg.wake_word}'")
        while self.running:
            self._set_status("standby")
            text = self.voice.listen(timeout=12, phrase_limit=4, retries=0)
            if self.cfg.wake_word in text:
                self._awake = True
                if self.on_wake:
                    self.on_wake()
                self.tts.speak_sync(f"Yes, {self.M}. I am listening.")
                self._command_loop()

    # ── Active command loop ────────────────────────────────────────── #
    def _command_loop(self):
        strikes = 0
        limit   = self.cfg.idle_strikes_limit

        while self.running:
            self._set_status("listening")
            text = self.voice.listen(retries=2)

            if not text:
                strikes += 1
                if strikes >= limit:
                    self.tts.speak(
                        f"No input detected. Going to standby, {self.M}. "
                        "Say Jarvis to wake me.")
                    self._awake = False
                    return
                continue

            strikes = 0

            # Sleep triggers
            if any(t in text for t in SLEEP_WORDS):
                self.tts.speak(
                    f"Going to standby. Call my name whenever you need me, {self.M}.")
                self._awake = False
                return

            # Hand off to AI executor
            self._dispatch(text)

    # ── Dispatch to AI ─────────────────────────────────────────────── #
    def _dispatch(self, raw: str):
        self._set_status("processing")

        response = self.executor.process(raw)

        # Handle multi-turn note sentinel
        if response == "__AWAIT_NOTE__":
            self.tts.speak_sync(
                f"What would you like me to write, {self.M}? Go ahead.")
            self._set_status("listening")
            content  = self.voice.listen(timeout=15, phrase_limit=20, retries=1)
            response = self.executor.save_note_content(content)

        self.tts.speak(response)

    # ── Listen helper (used by executor for multi-turn) ────────────── #
    def _listen_input(self) -> str:
        return self.voice.listen(timeout=15, phrase_limit=20, retries=1)

    # ── Status ─────────────────────────────────────────────────────── #
    def _set_status(self, status: str):
        if self.on_status:
            try:
                self.on_status(status)
            except Exception:
                pass

    # ── Stop ───────────────────────────────────────────────────────── #
    def stop(self):
        self.running = False
        self.voice.stop()
        self.tts.speak_sync(f"Shutting down. Goodbye, {self.M}.")
        self.tts.stop()
