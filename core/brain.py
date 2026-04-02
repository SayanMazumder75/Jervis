"""
core/brain.py  —  JARVIS Brain
Orchestrates wake-word loop, voice, AI executor,
screen vision, and audio capture.
"""
import threading
import logging
from typing import Callable, Optional

from core.voice         import TTSEngine, VoiceEngine
from core.executor      import AIExecutor
from core.screen_vision import (ScreenVision, classify_screen_command,
                                 extract_screen_question)
from core.audio_capture import (AudioCapture, classify_audio_command,
                                 extract_duration)

log = logging.getLogger("jarvis")

SLEEP_WORDS = ("go to sleep", "standby", "goodbye", "bye",
               "sleep now", "that will be all", "shut up")


class JarvisBrain:
    def __init__(self, cfg, existing_tts=None, face_auth=None):
        self.cfg = cfg
        self.M   = cfg.master_name

        # ── Voice systems ───────────────────────────────────────────── #
        self.tts = existing_tts or TTSEngine(
            rate        = cfg.tts_rate,
            volume      = cfg.tts_volume,
            voice_index = cfg.tts_voice_index,
        )
        self.voice = VoiceEngine(cfg)

        # ── AI executor ─────────────────────────────────────────────── #
        self.executor = AIExecutor(cfg, self._listen_input)

        # ── Screen Vision ────────────────────────────────────────────── #
        self.screen = ScreenVision()
        if self.screen.available:
            log.info("Screen vision: ENABLED  (Gemini)")
        else:
            log.warning("Screen vision: DISABLED  —  set GEMINI_API_KEY to enable")

        # ── Audio Capture ────────────────────────────────────────────── #
        self.audio = AudioCapture()
        if self.audio.available:
            log.info("Audio capture: ENABLED  (Whisper local)")
        else:
            log.warning("Audio capture: DISABLED  —  "
                        "pip install pyaudiowpatch openai-whisper")

        # ── Wire status events → HUD ─────────────────────────────────── #
        self.tts.on_start  = lambda: self._set_status("speaking")
        self.tts.on_end    = lambda: self._set_status(
            "listening" if self._awake else "standby")
        self.voice.status_cb = self._set_status

        # ── HUD callbacks (injected after construction) ──────────────── #
        self.on_status: Optional[Callable[[str], None]] = None
        self.on_wake:   Optional[Callable[[], None]]    = None

        self._awake  = False
        self.running = False

    # ════════════════════════════════════════════════════════════════════
    # Start / Boot
    # ════════════════════════════════════════════════════════════════════
    def start(self):
        self.running = True
        threading.Thread(target=self._boot, daemon=True, name="JARVIS").start()

    def _boot(self):
        self._set_status("calibrating")
        self.voice.calibrate()

        features = []
        if self.screen.available: features.append("screen vision")
        if self.audio.available:  features.append("audio capture")

        feat_str = (", ".join(features) + " active. ") if features else ""

        if self.executor.ai_available:
            self.tts.speak_sync(
                f"J.A.R.V.I.S online. Good day, {self.M}. "
                f"AI systems fully operational. {feat_str}"
                "Say my name to activate.")
        else:
            self.tts.speak_sync(
                f"J.A.R.V.I.S online. Good day, {self.M}. "
                "Note: AI key not detected. Running in basic mode. "
                "Say my name to activate.")
        self._wake_loop()

    # ════════════════════════════════════════════════════════════════════
    # Wake-word standby loop
    # ════════════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════════════
    # Active command loop
    # ════════════════════════════════════════════════════════════════════
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

            # ── Sleep ────────────────────────────────────────────────── #
            if any(t in text for t in SLEEP_WORDS):
                self.tts.speak(
                    f"Going to standby. Call my name whenever you need me, {self.M}.")
                self._awake = False
                return

            # ── Screen vision ─────────────────────────────────────────── #
            screen_cmd = classify_screen_command(text)
            if screen_cmd:
                self._handle_screen(text, screen_cmd)
                continue

            # ── Audio capture ─────────────────────────────────────────── #
            audio_cmd = classify_audio_command(text)
            if audio_cmd:
                self._handle_audio(text, audio_cmd)
                continue

            # ── AI executor (everything else) ─────────────────────────── #
            self._dispatch(text)

    # ════════════════════════════════════════════════════════════════════
    # Screen Vision Handler
    # ════════════════════════════════════════════════════════════════════
    def _handle_screen(self, text: str, cmd: str):
        if not self.screen.available:
            self.tts.speak(
                "Screen vision is not available. "
                "Please set your Gemini API key and install the required packages.")
            return

        self._set_status("processing")
        self.tts.speak_sync(f"Looking at your screen now, {self.M}.")

        # If user appended a custom question, use it
        custom_q = extract_screen_question(text)

        if custom_q:
            response = self.screen.analyze(custom_q)
        elif cmd == "read":
            response = self.screen.read_text()
        elif cmd == "answer":
            response = self.screen.answer_on_screen()
        elif cmd == "describe":
            response = self.screen.describe()
        else:
            response = self.screen.analyze()

        self.tts.speak(response)

    # ════════════════════════════════════════════════════════════════════
    # Audio Capture Handler
    # ════════════════════════════════════════════════════════════════════
    def _handle_audio(self, text: str, cmd: str):
        if not self.audio.available:
            self.tts.speak(
                "Audio capture is not available. "
                "Please install pyaudiowpatch and openai-whisper.")
            return

        duration = extract_duration(text)
        self._set_status("processing")

        dur_str = f"{int(duration)} seconds" if duration < 60 else f"{duration/60:.0f} minutes"

        if cmd == "mic":
            self.tts.speak_sync(
                f"Recording your microphone for {dur_str}, {self.M}.")
            response = self.audio.capture_mic_audio(duration)

        elif cmd == "mixed":
            self.tts.speak_sync(
                f"Recording microphone and system audio together for {dur_str}, {self.M}.")
            response = self.audio.capture_mixed_audio(duration)

        elif cmd == "video":
            self.tts.speak_sync(
                f"Listening to the video audio for {dur_str}, {self.M}.")
            response = self.audio.capture_video_audio(duration)

        else:  # "system" — default
            self.tts.speak_sync(
                f"Listening to system audio for {dur_str}, {self.M}.")
            response = self.audio.capture_system_audio(duration)

        if response:
            self.tts.speak(f"Here is what I heard: {response}")
        else:
            self.tts.speak(f"I could not transcribe any audio, {self.M}.")

    # ════════════════════════════════════════════════════════════════════
    # Dispatch to AI executor
    # ════════════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════════════
    # Helpers
    # ════════════════════════════════════════════════════════════════════
    def _listen_input(self) -> str:
        return self.voice.listen(timeout=15, phrase_limit=20, retries=1)

    def _set_status(self, status: str):
        if self.on_status:
            try:
                self.on_status(status)
            except Exception:
                pass

    def stop(self):
        self.running = False
        self.voice.stop()
        self.tts.speak_sync(f"Shutting down. Goodbye, {self.M}.")
        self.tts.stop()