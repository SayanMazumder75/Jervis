import speech_recognition as sr
import pyttsx3, threading, queue, time, logging

log = logging.getLogger("jarvis")


class TTSEngine:
    """Dedicated TTS thread — speak() is non-blocking."""

    def __init__(self, rate=160, volume=1.0, voice_index=0):
        self._q        = queue.Queue()
        self._done     = threading.Event()
        self._stop     = threading.Event()
        self._rate     = rate
        self._volume   = volume
        self._vidx     = voice_index
        self._speaking = False
        self.on_start  = None
        self.on_end    = None
        threading.Thread(target=self._run, daemon=True, name="TTS").start()

    def _run(self):
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        if voices and self._vidx < len(voices):
            engine.setProperty("voice", voices[self._vidx].id)
        engine.setProperty("rate",   self._rate)
        engine.setProperty("volume", self._volume)
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            if text is None:
                break
            self._speaking = True
            self._done.clear()
            if self.on_start:
                self.on_start()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                log.error(f"TTS: {e}")
            self._speaking = False
            self._done.set()
            if self.on_end:
                self.on_end()

    def speak(self, text: str):
        if not text:
            return
        log.info(f"JARVIS: {text}")
        self._q.put(text)

    def speak_sync(self, text: str, timeout=30.0):
        self.speak(text)
        time.sleep(0.1)
        self._done.wait(timeout=timeout)

    @property
    def is_speaking(self):
        return self._speaking

    def stop(self):
        self._stop.set()
        self._q.put(None)


class VoiceEngine:
    """Mic capture with calibration + retry."""

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
