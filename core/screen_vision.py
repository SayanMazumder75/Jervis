"""
core/screen_vision.py  —  JARVIS Screen Vision
========================================================
Takes a screenshot and uses Google Gemini Vision to:
  • Describe what's on screen
  • Read / OCR all text on screen
  • Answer any question visible on screen

Setup
-----
  1. Get a FREE Gemini API key: https://aistudio.google.com/app/apikey
  2. Set it:
       Windows CMD:  setx GEMINI_API_KEY "AIzaSyDIeJqDagmunVdTDvjbQtkFJJOZ7mC39Z8"
       Then restart terminal and run main.py

Install
-------
  pip install google-generativeai pillow pyautogui

Voice triggers (say after "Jarvis"):
  "what's on my screen"
  "look at my screen"
  "read the screen"
  "describe my screen"
  "answer this"   (for questions/problems on screen)
  "what does it say"
  "look at this and tell me [anything]"
"""

from __future__ import annotations

import os
import re
import logging
import tempfile
from typing import Optional

log = logging.getLogger("jarvis")

# ── Optional deps (graceful degradation) ────────────────────────────────
try:
    import pyautogui
    _PYAUTOGUI = True
except ImportError:
    _PYAUTOGUI = False

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False

try:
    from google import genai
    _GENAI = True
except ImportError:
    _GENAI = False


# ════════════════════════════════════════════════════════════════════════
class ScreenVision:
    """Capture screen → send to Gemini Vision → speak the answer."""

    MODEL = "gemini-1.5-flash"   # free-tier, fast, supports images

    def __init__(self):
        self._model: Optional[object] = None
        self.provider = self._init()

    # ── Init ────────────────────────────────────────────────────────────
    def _init(self) -> Optional[str]:
        if not _PYAUTOGUI:
            log.warning("ScreenVision: pyautogui not installed  →  pip install pyautogui")
            return None
        if not _PIL:
            log.warning("ScreenVision: Pillow not installed  →  pip install pillow")
            return None
        if not _GENAI:
            log.warning("ScreenVision: google-generativeai not installed  →  "
                        "pip install google-generativeai")
            return None

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            log.warning("ScreenVision: GEMINI_API_KEY not set  →  "
                        "get a free key at https://aistudio.google.com/app/apikey")
            return None

        try:
            self._client = genai.Client(api_key=key)
            self.MODEL = "gemini-2.0-flash"
            log.info(f"ScreenVision: ready  (Gemini {self.MODEL})")
            return "gemini"
        except Exception as e:
            log.error(f"ScreenVision init failed: {e}")
            return None

    @property
    def available(self) -> bool:
        return self.provider is not None

    # ── Screenshot ──────────────────────────────────────────────────────
    def _screenshot(self) -> Optional[Image.Image]:
        try:
            img = pyautogui.screenshot()
            # Downscale to max 1280×720 — keeps Gemini fast & cheap
            img.thumbnail((1280, 720), Image.LANCZOS)
            return img
        except Exception as e:
            log.error(f"Screenshot failed: {e}")
            return None

    # ── Ask Gemini ──────────────────────────────────────────────────────
    def _ask(self, img: Image.Image, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.MODEL,
                contents=[prompt, img]
            )
            return _clean_for_speech(response.text.strip())
        except Exception as e:
            log.error(f"Gemini vision error: {e}")
            return f"Screen analysis failed: {e}"

    # ── Public API ──────────────────────────────────────────────────────
    def analyze(self, question: str = "") -> str:
        """General: answer any question about the current screen."""
        if not self.available:
            return self._unavailable_msg()
        img = self._screenshot()
        if img is None:
            return "I could not capture the screen."
        prompt = _prompt_analyze(question)
        return self._ask(img, prompt)

    def describe(self) -> str:
        """Describe what application / content is on screen."""
        if not self.available:
            return self._unavailable_msg()
        img = self._screenshot()
        if img is None:
            return "I could not capture the screen."
        return self._ask(img, _PROMPT_DESCRIBE)

    def read_text(self) -> str:
        """OCR-style: read all visible text on screen."""
        if not self.available:
            return self._unavailable_msg()
        img = self._screenshot()
        if img is None:
            return "I could not capture the screen."
        return self._ask(img, _PROMPT_READ)

    def answer_on_screen(self) -> str:
        """Find and answer any question/problem visible on screen."""
        if not self.available:
            return self._unavailable_msg()
        img = self._screenshot()
        if img is None:
            return "I could not capture the screen."
        return self._ask(img, _PROMPT_ANSWER)

    def _unavailable_msg(self) -> str:
        parts = []
        if not _PYAUTOGUI:
            parts.append("pyautogui (pip install pyautogui)")
        if not _PIL:
            parts.append("Pillow (pip install pillow)")
        if not _GENAI:
            parts.append("google-generativeai (pip install google-generativeai)")
        if not os.environ.get("GEMINI_API_KEY"):
            parts.append("GEMINI_API_KEY environment variable")
        if parts:
            return ("Screen vision is not available. Missing: "
                    + ", and ".join(parts) + ".")
        return "Screen vision is not available."


# ── Prompts ──────────────────────────────────────────────────────────────
_BASE = (
    "You are JARVIS, an Iron Man-style AI assistant. "
    "The user is looking at their computer screen. "
    "Respond conversationally in 1 to 3 short sentences, as if speaking aloud. "
    "No markdown, no bullet points, no formatting — plain spoken English only."
)

_PROMPT_DESCRIBE = (
    f"{_BASE} "
    "Describe what is currently on this screen. "
    "Mention the application or website name and the key content visible."
)

_PROMPT_READ = (
    f"{_BASE} "
    "Read and transcribe all the text visible on this screen, from top to bottom. "
    "If there is a lot of text, summarise the key parts in spoken form."
)

_PROMPT_ANSWER = (
    f"{_BASE} "
    "Look at this screen carefully. "
    "If you see a question, math problem, coding error, or any task that needs an answer — "
    "answer it directly and concisely. "
    "If it is just content like an article or video, summarise the main point."
)


def _prompt_analyze(question: str) -> str:
    if question:
        return (
            f"{_BASE} "
            f"The user is asking about their screen: '{question}'. "
            "Answer that specific question based on what you see."
        )
    return _PROMPT_DESCRIBE


def _clean_for_speech(text: str) -> str:
    """Strip markdown so it sounds natural when spoken aloud."""
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)   # bold / italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)         # inline code / blocks
    text = re.sub(r"#{1,6}\s+", "", text)                  # headers
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.M)  # bullets
    text = re.sub(r"\n{2,}", " ", text)
    text = text.replace("\n", ". ").strip()
    # collapse multiple spaces/periods
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r" {2,}", " ", text)
    return text


# ── Trigger classification ────────────────────────────────────────────────
_DESCRIBE_KW = (
    "what's on my screen", "what is on my screen", "describe my screen",
    "describe the screen", "look at my screen", "look at the screen",
    "what am i looking at", "check my screen", "see my screen",
    "what is this", "screen vision",
)
_READ_KW = (
    "read the screen", "read my screen", "read text on screen",
    "read everything", "transcribe the screen", "read out the screen",
    "what does it say", "what does the screen say",
)
_ANSWER_KW = (
    "answer this", "solve this", "help me with this",
    "answer the question", "solve the problem", "what's the answer",
    "answer what's on", "help with screen",
)
_ANALYZE_KW = (
    "look at this and", "analyze my screen", "analyse my screen",
    "tell me about my screen", "what do you see",
)


def classify_screen_command(text: str) -> Optional[str]:
    """
    Returns one of: 'describe' | 'read' | 'answer' | 'analyze' | None
    None means the text is NOT a screen-vision command.
    """
    t = text.lower()
    if any(k in t for k in _READ_KW):    return "read"
    if any(k in t for k in _ANSWER_KW):  return "answer"
    if any(k in t for k in _DESCRIBE_KW): return "describe"
    if any(k in t for k in _ANALYZE_KW):  return "analyze"
    return None


def extract_screen_question(text: str) -> str:
    """
    Pull out a freeform question appended to a screen trigger.
    e.g. "look at my screen and tell me the author" → "tell me the author"
    """
    t = text.lower()
    for connector in (" and tell me ", " and say ", " and explain ",
                      " and what is ", " and who is ", " and how ",
                      " tell me ", " explain ", " what is "):
        if connector in t:
            idx = t.index(connector) + len(connector)
            return text[idx:].strip()
    return ""