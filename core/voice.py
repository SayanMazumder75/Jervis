"""
core/voice.py — Voice I/O
TTSEngine  — Reliable pyttsx3 wrapper, re-initializes engine after each use
VoiceEngine — Microphone capture with calibration + retry
"""

import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import logging

log = logging.getLogger("jarvis")


# ════════════════════════════════════════════════════════════════════════ #
#  TTS Engine  — reinitializes pyttsx3 for every utterance (most reliable)
# ════════════════════════════════════════════════════════════════════════ #
class TTSEngine:
    """
    Spawns a fresh pyttsx3 engine for EVERY speak call.
    This is the most reliable approach on Windows — avoids the engine
    getting stuck after the first runAndWait() call.
    """

    def __init__(self, rate=160, volume=1.0, voice_index=0):
        self._rate       = rate
        self._volume     = volume
        self._vidx       = voice_index
        self._lock       = threading.Lock()   # one speech at a time
        self._speaking   = False
        self.on_start    = None
        self.on_end      = None

        # Pre-detect voice ID once at startup
        self._voice_id   = self._detect_voice()

    def _detect_voice(self):
        """Get voice ID once at startup."""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            engine.stop()
            if voices and self._vidx < len(voices):
                return voices[self._vidx].id
            return None
        except Exception:
            return None

    def _speak_blocking(self, text: str):
        """Create fresh engine, speak, destroy. Called in thread."""
        try:
            engine = pyttsx3.init()
            if self._voice_id:
                engine.setProperty("voice",  self._voice_id)
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            log.error(f"TTS error: {e}")

    # ── Public API ──────────────────────────────────────────────────── #
    def speak(self, text: str):
        """Non-blocking: run speech in background thread."""
        if not text or not text.strip():
            return
        log.info(f"JARVIS: {text}")

        def _run():
            with self._lock:
                self._speaking = True
                if self.on_start:
                    try: self.on_start()
                    except: pass
                self._speak_blocking(text)
                self._speaking = False
                if self.on_end:
                    try: self.on_end()
                    except: pass

        threading.Thread(target=_run, daemon=True, name="TTS").start()

    def speak_sync(self, text: str, timeout: float = 30.0):
        """Blocking: speak and wait until done."""
        if not text or not text.strip():
            return
        log.info(f"JARVIS: {text}")

        done = threading.Event()

        def _run():
            with self._lock:
                self._speaking = True
                if self.on_start:
                    try: self.on_start()
                    except: pass
                self._speak_blocking(text)
                self._speaking = False
                if self.on_end:
                    try: self.on_end()
                    except: pass
                done.set()

        threading.Thread(target=_run, daemon=True, name="TTS-sync").start()
        done.wait(timeout=timeout)

    @property
    def is_speaking(self):
        return self._speaking

    def stop(self):
        pass   # nothing persistent to stop


# ════════════════════════════════════════════════════════════════════════ #
#  Voice (Mic) Engine
# ════════════════════════════════════════════════════════════════════════ #
class VoiceEngine:
    """Microphone capture with calibration and retry."""

    def __init__(self, cfg):
        self.cfg        = cfg
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold          = cfg.pause_threshold
        self.recognizer.energy_threshold         = cfg.energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.status_cb  = None
        self.running    = True

    def calibrate(self):
        try:
            self._set("calibrating")
            with sr.Microphone() as src:
                log.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(src, duration=2)
            log.info(f"Energy threshold set to {self.recognizer.energy_threshold:.0f}")
        except Exception as e:
            log.error(f"Calibration: {e}")
        finally:
            self._set("standby")

    def listen(self, timeout=None, phrase_limit=None, retries=2) -> str:
        timeout      = timeout      or self.cfg.listen_timeout
        phrase_limit = phrase_limit or self.cfg.phrase_time_limit

        for attempt in range(retries + 1):
            try:
                with sr.Microphone() as src:
                    self._set("listening")
                    try:
                        audio = self.recognizer.listen(
                            src, timeout=timeout,
                            phrase_time_limit=phrase_limit)
                    except sr.WaitTimeoutError:
                        return ""
                self._set("processing")
                text = self.recognizer.recognize_google(audio).lower().strip()
                log.info(f"Heard: '{text}'")
                return text
            except sr.UnknownValueError:
                if attempt < retries:
                    time.sleep(0.2)
                else:
                    return ""
            except sr.RequestError as e:
                log.error(f"Speech API: {e}")
                return ""
            except Exception as e:
                log.error(f"Listen: {e}")
                return ""
        return ""

    def _set(self, s):
        if self.status_cb:
            try:
                self.status_cb(s)
            except Exception:
                pass

    def stop(self):
        self.running = False