"""
core/audio_capture.py  —  JARVIS System Audio & Mixed Audio Capture
====================================================================
Captures THREE types of audio and transcribes them with Whisper:

  1. System audio   — what's playing on your PC speakers (loopback)
  2. Mic + system   — your voice mixed with what's playing
  3. Video/file     — whatever is audible while a video plays on screen

Uses PyAudioWPatch (Windows WASAPI loopback) + OpenAI Whisper (runs locally, FREE).

Install
-------
  pip install pyaudiowpatch openai-whisper numpy

  Note: openai-whisper requires ffmpeg on PATH:
    Windows: https://ffmpeg.org/download.html  → add bin/ to PATH

Voice triggers (say after "Jarvis"):
  "what is playing"
  "transcribe the audio"
  "what are they saying"
  "listen to the video"
  "capture audio for 20 seconds"   ← custom duration
  "what was said"
  "transcribe this video"
"""

from __future__ import annotations

import io
import logging
import os
import queue
import re
import tempfile
import threading
import time
import wave
from typing import Optional, Tuple

log = logging.getLogger("jarvis")

# ── Optional deps (graceful degradation) ────────────────────────────────
try:
    import pyaudiowpatch as pyaudio   # Windows WASAPI loopback support
    _WPATCH = True
except ImportError:
    try:
        import pyaudio                # fallback: mic only
        _WPATCH = False
        log.warning("AudioCapture: pyaudiowpatch not found, falling back to "
                    "standard PyAudio (mic only). "
                    "For system audio: pip install pyaudiowpatch")
    except ImportError:
        pyaudio = None
        _WPATCH = False

try:
    import whisper as _whisper_lib
    _WHISPER = True
except ImportError:
    _whisper_lib = None
    _WHISPER = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# ════════════════════════════════════════════════════════════════════════
class AudioCapture:
    """
    Record system audio, microphone, or both — then transcribe with Whisper.
    All processing is local (no API cost).
    """

    CHUNK        = 1024
    FORMAT_INT16 = None      # set after pyaudio import
    CHANNELS     = 2
    RATE         = 44100
    MAX_DURATION = 30.0      # hard cap to avoid huge recordings
    DEFAULT_DUR  = 10.0

    def __init__(self):
        if pyaudio is not None:
            self.FORMAT_INT16 = pyaudio.paInt16
        self._model      = None          # Whisper model (lazy load)
        self._model_lock = threading.Lock()
        self.available   = self._check()

    # ── Availability ────────────────────────────────────────────────────
    def _check(self) -> bool:
        if pyaudio is None:
            log.warning("AudioCapture: pyaudio not installed  →  "
                        "pip install pyaudiowpatch")
            return False
        if not _WHISPER:
            log.warning("AudioCapture: openai-whisper not installed  →  "
                        "pip install openai-whisper")
            return False
        if not _NUMPY:
            log.warning("AudioCapture: numpy not installed  →  pip install numpy")
            return False
        log.info(f"AudioCapture: ready  (wpatch={_WPATCH})")
        return True

    # ── Lazy-load Whisper ────────────────────────────────────────────────
    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                log.info("Loading Whisper 'base' model — first load ~20 s …")
                self._model = _whisper_lib.load_model("base")
                log.info("Whisper model ready.")
        return self._model

    # ── Find devices ─────────────────────────────────────────────────────
    @staticmethod
    def _find_loopback(pa) -> Tuple[Optional[int], int]:
        """Return (device_index, sample_rate) for the WASAPI loopback device."""
        if not _WPATCH:
            return None, AudioCapture.RATE   # mic only fallback

        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            rate = int(default_out["defaultSampleRate"])

            # Is the default output device already a loopback?
            if default_out.get("isLoopbackDevice", False):
                return default_out["index"], rate

            # Search for its loopback twin
            for i in range(pa.get_device_count()):
                dev = pa.get_device_info_by_index(i)
                if (dev.get("isLoopbackDevice", False) and
                        default_out["name"] in dev["name"]):
                    log.info(f"Loopback device: {dev['name']} @ {rate} Hz")
                    return i, rate

            # Use default output as loopback (WASAPI will provide it)
            return default_out["index"], rate

        except Exception as e:
            log.error(f"_find_loopback: {e}")
            return None, AudioCapture.RATE

    @staticmethod
    def _find_microphone(pa) -> Tuple[Optional[int], int]:
        """Return (device_index, sample_rate) for the default input device."""
        try:
            idx = pa.get_default_input_device_info()["index"]
            rate = int(pa.get_device_info_by_index(idx)["defaultSampleRate"])
            return idx, rate
        except Exception as e:
            log.error(f"_find_microphone: {e}")
            return None, 44100

    # ── Core record function ──────────────────────────────────────────────
    def _record_device(self, device_index: Optional[int],
                       sample_rate: int, channels: int,
                       duration: float) -> bytes:
        """Record `duration` seconds from a specific device. Returns raw PCM bytes."""
        pa = pyaudio.PyAudio()
        frames = []
        try:
            kwargs = dict(
                format=self.FORMAT_INT16,
                channels=channels,
                rate=sample_rate,
                frames_per_buffer=self.CHUNK,
                input=True,
            )
            if device_index is not None:
                kwargs["input_device_index"] = device_index

            stream = pa.open(**kwargs)
            n_chunks = int(sample_rate / self.CHUNK * duration)
            for _ in range(n_chunks):
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            log.error(f"_record_device: {e}")
        finally:
            pa.terminate()
        return b"".join(frames)

    # ── Write WAV helper ─────────────────────────────────────────────────
    @staticmethod
    def _write_wav(pcm: bytes, channels: int, sample_rate: int) -> str:
        tmp = tempfile.mktemp(suffix=".wav", prefix="jarvis_audio_")
        pa = pyaudio.PyAudio()
        sample_width = pa.get_sample_size(pyaudio.paInt16)
        pa.terminate()
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return tmp

    # ── Mix two PCM streams ──────────────────────────────────────────────
    @staticmethod
    def _mix_pcm(pcm_a: bytes, pcm_b: bytes) -> bytes:
        """Mix two int16 PCM byte arrays (zero-pad shorter one)."""
        a = np.frombuffer(pcm_a, dtype=np.int16).astype(np.float32)
        b = np.frombuffer(pcm_b, dtype=np.int16).astype(np.float32)
        # Pad to same length
        if len(a) < len(b):
            a = np.pad(a, (0, len(b) - len(a)))
        elif len(b) < len(a):
            b = np.pad(b, (0, len(a) - len(b)))
        mixed = np.clip((a + b) / 2, -32768, 32767).astype(np.int16)
        return mixed.tobytes()

    # ── Transcribe WAV ────────────────────────────────────────────────────
    def _transcribe(self, wav_path: str) -> str:
        try:
            model = self._get_model()
            log.info(f"Whisper transcribing: {wav_path}")
            result = model.transcribe(wav_path, fp16=False)
            text = result.get("text", "").strip()
            log.info(f"Transcription: {text[:120]}")
            return text or "I couldn't make out any speech in that recording."
        except Exception as e:
            log.error(f"Whisper: {e}")
            return f"Transcription failed: {e}"
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════════

    def capture_system_audio(self, duration: float = DEFAULT_DUR) -> str:
        """
        Record what's playing on the PC speakers (loopback) and transcribe it.
        Works with videos, music, meetings — anything audible on screen.
        """
        if not self.available:
            return self._unavailable_msg()

        duration = min(duration, self.MAX_DURATION)
        pa = pyaudio.PyAudio()
        dev_idx, rate = self._find_loopback(pa)
        pa.terminate()

        log.info(f"Recording system audio ({duration:.0f}s) …")
        pcm = self._record_device(dev_idx, rate, channels=2, duration=duration)
        if not pcm:
            return "System audio recording failed — nothing was captured."

        wav = self._write_wav(pcm, channels=2, sample_rate=rate)
        return self._transcribe(wav)

    def capture_mic_audio(self, duration: float = DEFAULT_DUR) -> str:
        """Record microphone only and transcribe."""
        if not self.available:
            return self._unavailable_msg()

        duration = min(duration, self.MAX_DURATION)
        pa = pyaudio.PyAudio()
        dev_idx, rate = self._find_microphone(pa)
        pa.terminate()

        log.info(f"Recording microphone ({duration:.0f}s) …")
        pcm = self._record_device(dev_idx, rate, channels=1, duration=duration)
        if not pcm:
            return "Microphone recording failed."

        wav = self._write_wav(pcm, channels=1, sample_rate=rate)
        return self._transcribe(wav)

    def capture_mixed_audio(self, duration: float = DEFAULT_DUR) -> str:
        """
        Record microphone AND system audio simultaneously, mix them,
        then transcribe the combined audio.
        Perfect for capturing both sides of a conversation or
        your voice alongside video audio.
        """
        if not self.available:
            return self._unavailable_msg()
        if not _NUMPY:
            return "Mixed audio capture needs numpy: pip install numpy"

        duration = min(duration, self.MAX_DURATION)

        pa = pyaudio.PyAudio()
        lb_idx,  lb_rate  = self._find_loopback(pa)
        mic_idx, mic_rate = self._find_microphone(pa)
        pa.terminate()

        log.info(f"Recording mic + system audio mixed ({duration:.0f}s) …")

        # Record both in parallel threads
        results = {}

        def rec(key, idx, rate, ch):
            results[key] = (self._record_device(idx, rate, ch, duration), rate, ch)

        t1 = threading.Thread(target=rec, args=("sys", lb_idx,  lb_rate,  2))
        t2 = threading.Thread(target=rec, args=("mic", mic_idx, mic_rate, 1))
        t1.start(); t2.start()
        t1.join();  t2.join()

        sys_pcm, sys_rate, _ = results.get("sys", (b"", lb_rate,  2))
        mic_pcm, mic_rate, _ = results.get("mic", (b"", mic_rate, 1))

        if not sys_pcm and not mic_pcm:
            return "Mixed audio recording failed — nothing was captured."

        # Resample mic to sys rate if they differ (simple: use sys audio rate)
        # For simplicity we just use whichever has content; ideally they share 44100
        if sys_pcm and mic_pcm:
            mixed = self._mix_pcm(sys_pcm, mic_pcm)
            rate  = sys_rate
        elif sys_pcm:
            mixed = sys_pcm
            rate  = sys_rate
        else:
            mixed = mic_pcm
            rate  = mic_rate

        wav = self._write_wav(mixed, channels=1 if not sys_pcm else 2,
                              sample_rate=rate)
        return self._transcribe(wav)

    def capture_video_audio(self, duration: float = DEFAULT_DUR) -> str:
        """
        Alias for system audio capture, but named semantically for
        'transcribe what's in this video'.
        """
        return self.capture_system_audio(duration)

    # ── Unavailable message ──────────────────────────────────────────────
    def _unavailable_msg(self) -> str:
        missing = []
        if pyaudio is None:
            missing.append("pyaudiowpatch (pip install pyaudiowpatch)")
        if not _WHISPER:
            missing.append("openai-whisper (pip install openai-whisper)")
        if not _NUMPY:
            missing.append("numpy (pip install numpy)")
        return ("Audio capture is not available. Missing: "
                + ", ".join(missing) + ".")


# ── Trigger classification ────────────────────────────────────────────────

_SYS_KW = (
    "what is playing", "what's playing",
    "transcribe the audio", "transcribe audio",
    "what are they saying", "what is being said",
    "capture audio", "listen to audio",
    "hear the screen", "transcribe the video",
    "listen to the video", "what did they say",
    "transcribe this video", "transcribe this",
    "what was said",
)
_MIC_KW = (
    "record my voice", "capture my voice",
    "transcribe what i say", "transcribe me",
)
_MIX_KW = (
    "mix audio", "record both", "capture both",
    "mic and system", "microphone and speaker",
    "transcribe both sides",
)


def classify_audio_command(text: str) -> Optional[str]:
    """
    Returns: 'system' | 'mic' | 'mixed' | 'video' | None
    """
    t = text.lower()
    if any(k in t for k in _MIX_KW):   return "mixed"
    if any(k in t for k in _MIC_KW):   return "mic"
    if "video" in t and "transcrib" in t: return "video"
    if any(k in t for k in _SYS_KW):   return "system"
    return None


def extract_duration(text: str) -> float:
    """Parse 'for N seconds' or 'for N minutes' from command text."""
    t = text.lower()
    m = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(second|sec|minute|min)", t)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if "min" in unit:
            val = min(val * 60, AudioCapture.MAX_DURATION)
        return min(val, AudioCapture.MAX_DURATION)
    return AudioCapture.DEFAULT_DUR