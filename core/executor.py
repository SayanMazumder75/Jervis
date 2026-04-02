"""
core/executor.py  —  AI-Powered Command Executor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AI Backend Priority (auto-detected, no config needed):
  1. Google Gemini  — FREE  (set GEMINI_API_KEY)  ← recommended
  2. OpenAI GPT     — PAID  (set OPENAI_API_KEY)  ← fallback

Get free Gemini key: https://aistudio.google.com
Install:  pip install google-genai

Flow: voice text → AI → JSON action plan → executor runs action → JARVIS speaks
"""

import os, subprocess, webbrowser, socket, datetime, logging, json, re
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger("jarvis")

# ── System prompt sent to AI with every request ─────────────────────── #
SYSTEM_PROMPT = """You are J.A.R.V.I.S — Just A Rather Very Intelligent System.
You are an AI assistant running on Windows. The user speaks to you and you decide what to do.

Your job: analyze the user's request and respond with a JSON object ONLY. No extra text.

JSON format:
{
  "action": "<action_name>",
  "params": { ... },
  "speak": "<what JARVIS says aloud — conversational, witty, like Iron Man's JARVIS>"
}

Available actions and their params:
- open_website:    { "url": "https://..." }
- open_app:        { "app": "chrome|firefox|notepad|vscode|calculator|explorer|paint|taskmgr|spotify|word|excel|discord|steam|vlc|powershell|cmd|settings|control" }
- system_info:     { "type": "battery|cpu|ram|disk|ip|uptime|username" }
- web_search:      { "query": "search terms" }
- time_date:       { "type": "time|date|day|year|datetime" }
- system_control:  { "action": "shutdown|restart|sleep|lock|logoff|cancel_shutdown" }
- volume:          { "direction": "up|down|mute|unmute" }
- media_control:   { "action": "play_pause|next|previous" }
- create_note:     { "content": "note text here" }
- screenshot:      {}
- open_folder:     { "name": "desktop|downloads|documents|pictures|music|videos" }
- type_text:       { "text": "exact text to type", "mode": "type|enter" }
- speak_only:      {}

Rules:
- For ANY website: use open_website with the correct URL. Figure out the URL yourself.
- For ANY app: use open_app with the closest matching app name.
- If user asks "open X" where X is a website, pick open_website.
- If user asks a question (time, date, facts, jokes, calculations): use speak_only and answer in the speak field.
- If user wants system info: use system_info (the system will fill in real values for you — write speak as a template like "Battery is {battery_percent}% and {battery_status}" — use {placeholders} that match the type).
- For time/date: use time_date (system fills real values — use {time}, {date}, {day}, {year} placeholders in speak).
- Be witty, intelligent, and formal like JARVIS from Iron Man. Address user as {master}.
- Keep speak responses concise and suitable for text-to-speech (no markdown, no bullet points).
- If user says "type", "write", "dictate", "type this", "write this in", or asks to type something into an app: use type_text with the exact words to type. Set mode to "enter" if they want to send/submit after typing.
- ALWAYS return valid JSON only. Nothing else."""


class AIExecutor:
    """
    Auto-detects available AI backend (Gemini free / OpenAI paid).
    Sends user text to AI, gets JSON action plan, executes it locally.
    """

    def __init__(self, cfg, listen_fn: Optional[Callable] = None, tts=None):
        self.cfg      = cfg
        self.listen   = listen_fn
        self.M        = cfg.master_name
        self._history = []
        self._tts = tts   # direct TTS access to speak during waits

        # ── Key pools (supports multiple keys per provider) ──────── #
        # Reads: GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 ... (up to 10)
        # Reads: OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3 ... (up to 10)
        self._groq_keys    = self._load_keys("GROQ_API_KEY")
        self._gemini_keys  = self._load_keys("GEMINI_API_KEY")
        self._openai_keys  = self._load_keys("OPENAI_API_KEY")

        # Active key index per provider
        self._groq_idx     = 0
        self._gemini_idx   = 0
        self._openai_idx   = 0

        # Exhausted flags per key: {key: True/False}
        self._groq_exhausted   = {}
        self._gemini_exhausted = {}
        self._openai_exhausted = {}

        # Current built clients
        self._groq_clients   = {}
        self._gemini_models  = {}
        self._openai_clients = {}
        self._gemini_model_name = "gemini-1.5-flash-8b"

        self._backend = None
        self._setup()

    # ── Speak helper (speaks immediately during processing) ─────────── #
    def _say(self, text: str):
        """Speak immediately without waiting for dispatch loop."""
        if self._tts and text:
            self._tts.speak(text)

    # ── Load all numbered keys from env ───────────────────────────── #
    @staticmethod
    def _load_keys(prefix: str) -> list:
        keys = []
        # Always check the base key first (e.g. GEMINI_API_KEY)
        base = os.environ.get(prefix, "").strip()
        if base:
            keys.append(base)
        # Then numbered variants: GEMINI_API_KEY_2, _3 ... _10
        for i in range(2, 11):
            k = os.environ.get(f"{prefix}_{i}", "").strip()
            if k:
                keys.append(k)
        return keys

    # ── Build Gemini model for a specific key ─────────────────────── #
    def _build_gemini(self, api_key: str):
        """Build Gemini client using new google-genai SDK."""
        from google import genai as new_genai
        from google.genai import types as genai_types

        client = new_genai.Client(api_key=api_key)

        # Pick best available model
        CANDIDATES = [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        chosen = "gemini-2.0-flash-lite"
        try:
            available = [m.name for m in client.models.list()]
            avail_names = [a.split("/")[-1] if "/" in a else a for a in available]
            for c in CANDIDATES:
                if any(c in a for a in avail_names):
                    chosen = c
                    break
        except Exception:
            pass

        self._gemini_model_name = chosen
        self._genai_types = genai_types
        return client   # return client, model name stored separately

    # ── Setup: detect which backends are available ────────────────── #
    def _setup(self):
        # Priority 1: Groq (FREE, 14400 req/day, fastest)
        if self._groq_keys:
            try:
                from groq import Groq   # noqa
                self._backend = "groq"
                log.info(f"Groq keys loaded: {len(self._groq_keys)} key(s) — 14400 req/day free")
                return
            except ImportError:
                log.warning("groq not installed. Run: pip install groq")

        # Priority 2: Gemini (FREE, 1500 req/day)
        if self._gemini_keys:
            try:
                from google import genai as _chk   # noqa
                self._backend = "gemini"
                log.info(f"Gemini keys loaded: {len(self._gemini_keys)} account(s)")
                return
            except ImportError:
                log.warning("google-genai not installed. Run: pip install google-genai")

        # Priority 3: OpenAI (PAID)
        if self._openai_keys:
            try:
                from openai import OpenAI   # noqa
                self._backend = "openai"
                log.info(f"OpenAI keys loaded: {len(self._openai_keys)} account(s)")
                return
            except ImportError:
                log.error("openai not installed. Run: pip install openai")

        log.warning(
            "No AI keys found.\n"
            "  BEST (Free 14400/day): set GROQ_API_KEY at console.groq.com\n"
            "  Alt  (Free 1500/day):  set GEMINI_API_KEY at aistudio.google.com\n"
            "  Paid: set OPENAI_API_KEY at platform.openai.com")

    # ── Active model getters ───────────────────────────────────────── #
    def _active_gemini_model(self):
        """Return current Gemini model, building it if needed."""
        key = self._gemini_keys[self._gemini_idx]
        if key not in self._gemini_models:
            self._gemini_models[key] = self._build_gemini(key)
        return self._gemini_models[key], key

    def _active_openai_client(self):
        from openai import OpenAI
        key = self._openai_keys[self._openai_idx]
        if key not in self._openai_clients:
            self._openai_clients[key] = OpenAI(api_key=key)
        return self._openai_clients[key], key

    def _active_groq_client(self):
        from groq import Groq
        key = self._groq_keys[self._groq_idx]
        if key not in self._groq_clients:
            self._groq_clients[key] = Groq(api_key=key)
        return self._groq_clients[key], key

    # ── Rotate to next key ────────────────────────────────────────── #
    def _rotate_gemini(self, exhausted_key: str) -> bool:
        """Mark key as exhausted, try next. Returns True if a fresh key exists."""
        self._gemini_exhausted[exhausted_key] = True
        for i, key in enumerate(self._gemini_keys):
            if not self._gemini_exhausted.get(key, False):
                self._gemini_idx = i
                log.info(f"Rotated to Gemini key #{i+1}")
                return True
        log.warning("All Gemini keys exhausted.")
        return False

    def _rotate_openai(self, exhausted_key: str) -> bool:
        self._openai_exhausted[exhausted_key] = True
        for i, key in enumerate(self._openai_keys):
            if not self._openai_exhausted.get(key, False):
                self._openai_idx = i
                log.info(f"Rotated to OpenAI key #{i+1}")
                return True
        log.warning("All OpenAI keys exhausted.")
        return False

    def _rotate_groq(self, exhausted_key: str) -> bool:
        self._groq_exhausted[exhausted_key] = True
        for i, key in enumerate(self._groq_keys):
            if not self._groq_exhausted.get(key, False):
                self._groq_idx = i
                log.info(f"Rotated to Groq key #{i+1}")
                return True
        log.warning("All Groq keys exhausted.")
        return False

    # ── Reset exhausted flags (called after sufficient wait) ──────── #
    def _reset_exhausted(self):
        self._groq_exhausted.clear()
        self._gemini_exhausted.clear()
        self._openai_exhausted.clear()
        self._groq_idx   = 0
        self._gemini_idx = 0
        self._openai_idx = 0
        log.info("API key quotas reset — starting from key #1")

    @property
    def ai_available(self):
        return self._backend is not None

    @property
    def backend_name(self):
        if self._backend == "groq":
            idx   = self._groq_idx + 1
            total = len(self._groq_keys)
            return f"Groq Free (key {idx}/{total})"
        if self._backend == "gemini":
            idx   = self._gemini_idx + 1
            total = len(self._gemini_keys)
            return f"Gemini Free (key {idx}/{total})"
        if self._backend == "openai":
            idx   = self._openai_idx + 1
            total = len(self._openai_keys)
            return f"OpenAI (key {idx}/{total})"
        return "Offline"

    # ── Main entry point ──────────────────────────────────────────── #
    def process(self, user_text: str) -> str:
        """
        Smart routing:
        1. Try local handler first (time, date, greetings, jokes, system info)
           → Zero API calls, instant response
        2. Only call AI for things that truly need it
        """
        # Step 1: try local handler — saves API quota
        local = self._local_handler(user_text)
        if local:
            return local

        # Step 2: needs AI
        if not self.ai_available:
            return self._no_ai_fallback(user_text)
        if self._backend == "groq":
            return self._process_groq(user_text)
        elif self._backend == "gemini":
            return self._process_gemini(user_text)
        else:
            return self._process_openai(user_text)

    # ── Local handler — zero API calls ────────────────────────────── #
    def _local_handler(self, text: str) -> str:
        """
        Handle common commands locally without touching the API.
        Returns response string if handled, empty string if AI is needed.
        """
        import random
        t = text.lower().strip()

        # ── Time ─────────────────────────────────────────────────── #
        if any(w in t for w in ["what time","current time","what is the time",
                                 "tell me the time","what's the time"]):
            now = datetime.datetime.now()
            return f"The current time is {now:%I:%M %p}, {self.M}."

        # ── Date ─────────────────────────────────────────────────── #
        if any(w in t for w in ["what date","today's date","what is the date",
                                 "what day is it","current date","what's the date"]):
            return f"Today is {datetime.datetime.now():%A, %B %d, %Y}."

        # ── Greetings ─────────────────────────────────────────────── #
        if any(w in t for w in ["hello","hi jarvis","good morning",
                                 "good afternoon","good evening","hey jarvis"]):
            h = datetime.datetime.now().hour
            g = "Good morning" if h<12 else ("Good afternoon" if h<17 else "Good evening")
            return f"{g}, {self.M}. All systems are online. How can I help you?"

        # ── How are you ───────────────────────────────────────────── #
        if any(w in t for w in ["how are you","you okay","you good","system status"]):
            return (f"All systems running at optimal capacity, {self.M}. "
                    "No anomalies detected.")

        # ── Jokes ─────────────────────────────────────────────────── #
        if any(w in t for w in ["tell me a joke","joke","make me laugh"]):
            jokes = [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "There are only 10 types of people — those who understand binary and those who don't.",
                "I would tell you a joke about UDP, but you might not get it.",
                "A SQL query walks into a bar, walks up to two tables and asks: may I join you?",
                "Why did the developer go broke? He used up all his cache.",
                "Debugging is like being the detective in a crime film where you are also the murderer.",
            ]
            return random.choice(jokes)

        # ── Fun facts ─────────────────────────────────────────────── #
        if any(w in t for w in ["fun fact","interesting fact","tell me something"]):
            facts = [
                "Honey never spoils. Archaeologists found 3000-year-old honey in Egyptian tombs that was still edible.",
                "A day on Venus is longer than a year on Venus.",
                "Octopuses have three hearts, blue blood, and can edit their own RNA.",
                "Bananas are berries but strawberries are not.",
                "The human brain uses about 20 watts — enough to power a dim light bulb.",
            ]
            return random.choice(facts)

        # ── Coin / Dice ───────────────────────────────────────────── #
        if any(w in t for w in ["flip a coin","coin toss","heads or tails"]):
            return f"I flipped a coin. It is {random.choice(['Heads','Tails'])}."
        if any(w in t for w in ["roll a dice","roll dice","roll the dice"]):
            return f"I rolled a six-sided die. The result is {random.randint(1,6)}."

        # ── Meaning of life ───────────────────────────────────────── #
        if "meaning of life" in t:
            return ("The answer is 42. At least according to Deep Thought. "
                    "Though I suspect it involves good code and a stable power supply.")

        # ── Who are you ───────────────────────────────────────────── #
        if any(w in t for w in ["who are you","what are you","introduce yourself"]):
            return (f"I am J.A.R.V.I.S — Just A Rather Very Intelligent System, {self.M}. "
                    "Your personal AI assistant. I can control your computer, "
                    "open anything, answer questions, all by voice.")

        # ── Help ──────────────────────────────────────────────────── #
        if any(w in t for w in ["what can you do","help","commands"]):
            return (f"I can open any website or app, search Google, tell you the time and date, "
                    f"check battery, CPU, RAM and disk, control volume, "
                    f"take screenshots, create notes, and answer any question using AI. "
                    f"Just speak naturally, {self.M}. I understand everything.")

        # ── Battery ───────────────────────────────────────────────── #
        if "battery" in t:
            try:
                import psutil
                b = psutil.sensors_battery()
                if b:
                    s = "charging" if b.power_plugged else "discharging"
                    return f"Battery is at {b.percent:.0f} percent and {s}, {self.M}."
                return "This device does not report battery information."
            except: pass

        # ── CPU ───────────────────────────────────────────────────── #
        if any(w in t for w in ["cpu usage","processor usage","cpu"]):
            try:
                import psutil
                c = psutil.cpu_percent(interval=1)
                return f"CPU usage is {c:.0f} percent, {self.M}."
            except: pass

        # ── RAM ───────────────────────────────────────────────────── #
        if any(w in t for w in ["ram usage","memory usage","ram"]):
            try:
                import psutil
                m = psutil.virtual_memory()
                return (f"RAM is {m.percent:.0f} percent used. "
                        f"{m.available/1e9:.1f} gigabytes available of {m.total/1e9:.1f} total.")
            except: pass

        # ── Disk ──────────────────────────────────────────────────── #
        if any(w in t for w in ["disk space","disk usage","storage","hard drive"]):
            try:
                import psutil
                d = psutil.disk_usage("C:\\")
                return (f"C drive is {d.percent:.0f} percent full. "
                        f"{d.free/1e9:.1f} gigabytes free of {d.total/1e9:.1f} total.")
            except: pass

        # ── IP address ────────────────────────────────────────────── #
        if any(w in t for w in ["ip address","my ip","what is my ip"]):
            try:
                import socket
                ip = socket.gethostbyname(socket.gethostname())
                return f"Your local IP address is {ip}."
            except: pass

        # ── Volume (no AI needed) ─────────────────────────────────── #
        if any(w in t for w in ["volume up","increase volume","louder","turn up"]):
            self._do_volume({"direction":"up"}, "")
            return f"Volume increased, {self.M}."
        if any(w in t for w in ["volume down","decrease volume","quieter","turn down"]):
            self._do_volume({"direction":"down"}, "")
            return f"Volume decreased, {self.M}."
        if any(w in t for w in ["mute","unmute","toggle mute"]):
            self._do_volume({"direction":"mute"}, "")
            return "Audio toggled."

        # ── Screenshot (no AI needed) ─────────────────────────────── #
        if any(w in t for w in ["screenshot","take screenshot","capture screen"]):
            return self._do_screenshot(f"Screenshot saved to Desktop, {self.M}.")

        # ── Lock / Sleep (no AI needed) ───────────────────────────── #
        if any(w in t for w in ["lock","lock screen","lock computer","lock my computer"]):
            os.system("rundll32.exe user32.dll,LockWorkStation")
            return f"Locking workstation. Goodbye for now, {self.M}."
        if t in ["sleep","go to sleep","hibernate"]:
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
            return "Initiating sleep mode."

        # ── Open common websites without AI ───────────────────────── #
        QUICK_SITES = {
            "youtube": "https://www.youtube.com",
            "google":  "https://www.google.com",
            "github":  "https://www.github.com",
            "gmail":   "https://mail.google.com",
            "instagram":"https://www.instagram.com",
            "facebook":"https://www.facebook.com",
            "twitter": "https://www.twitter.com",
            "reddit":  "https://www.reddit.com",
            "netflix": "https://www.netflix.com",
            "whatsapp":"https://web.whatsapp.com",
            "linkedin":"https://www.linkedin.com",
            "amazon":  "https://www.amazon.com",
            "chatgpt": "https://chat.openai.com",
            "wikipedia":"https://www.wikipedia.org",
        }
        for site, url in QUICK_SITES.items():
            if f"open {site}" in t:
                webbrowser.open(url)
                return f"Opening {site.capitalize()}, {self.M}."

        # ── Open common apps without AI ────────────────────────────── #
        QUICK_APPS = {
            "chrome":      "chrome",
            "notepad":     "notepad.exe",
            "calculator":  "calc.exe",
            "explorer":    "explorer.exe",
            "task manager":"taskmgr.exe",
            "paint":       "mspaint.exe",
            "cmd":         "cmd.exe",
            "powershell":  "powershell.exe",
        }
        for app, cmd in QUICK_APPS.items():
            if f"open {app}" in t or f"launch {app}" in t:
                subprocess.Popen(cmd, shell=True)
                return f"Opening {app}, {self.M}."

        # ── Web search without AI ──────────────────────────────────── #
        for prefix in ["search for ","search google for ","google "]:
            if t.startswith(prefix):
                q = t[len(prefix):].strip()
                webbrowser.open(f"https://www.google.com/search?q={q.replace(' ','+')}")
                return f"Searching Google for {q}."

        # ── Type / dictate text ────────────────────────────────────── #
        # Triggers: "type hello", "write hello", "type this: hello"
        for prefix in ["type this ", "type ", "write this ", "dictate "]:
            if t.startswith(prefix):
                to_type = text[len(prefix):].strip()   # preserve original case
                if to_type:
                    return self._do_type_text({"text": to_type, "mode": "type"},
                                              f"Typing: {to_type}")

        # "start dictation" / "start typing" → multi-word listen mode
        if any(p in t for p in ["start dictation","start typing","begin dictation",
                                  "dictation mode","typing mode"]):
            return "__AWAIT_DICTATION__"

        # ── Face recognition commands ─────────────────────────────── #
        if any(p in t for p in ["register my face","register face",
                                  "add my face","setup face recognition"]):
            return "__REGISTER_FACE__"

        if any(p in t for p in ["delete face","remove face","reset face",
                                  "clear face data","forget my face"]):
            return "__DELETE_FACE__"

        if any(p in t for p in ["who am i","verify my face","check my face"]):
            return "__VERIFY_FACE__"

        # Not handled locally — let AI deal with it
        return ""

    # ── Groq backend (FREE, 14400 req/day, fastest) ─────────────────── #
    def _process_groq(self, user_text: str) -> str:
        import time
        GROQ_MODEL = "llama-3.3-70b-versatile"   # fast, smart, free

        keys_tried = 0
        total_keys = len(self._groq_keys)

        while keys_tried < total_keys:
            client, active_key = self._active_groq_client()
            keys_tried += 1

            try:
                system  = SYSTEM_PROMPT.replace("{master}", self.M)
                messages = [{"role": "system", "content": system}]
                for msg in self._history[-10:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": user_text})

                for attempt in range(2):
                    try:
                        resp = client.chat.completions.create(
                            model=GROQ_MODEL,
                            messages=messages,
                            max_tokens=400,
                            temperature=0.4,
                            response_format={"type": "json_object"},
                        )
                        raw = resp.choices[0].message.content.strip()
                        break
                    except Exception as e:
                        err = str(e)
                        if ("429" in err or "rate" in err.lower()
                                or "quota" in err.lower()):
                            if attempt == 0:
                                log.warning(f"Groq key #{self._groq_idx+1} rate limited. Retry 3s...")
                                time.sleep(3)
                            else:
                                raise
                        else:
                            raise

                # Clean JSON
                raw = re.sub(r"```json", "", raw)
                raw = re.sub(r"```",     "", raw).strip()
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)

                log.debug(f"Groq key #{self._groq_idx+1}: {raw[:80]}")
                plan   = json.loads(raw)
                spoken = self._execute(plan)

                self._history.append({"role": "user",      "content": user_text})
                self._history.append({"role": "assistant",  "content": raw})
                if len(self._history) > 14:
                    self._history = self._history[-14:]

                return spoken

            except json.JSONDecodeError:
                m = re.search(r'"speak"\s*:\s*"([^"]+)"', raw if "raw" in dir() else "")
                if m:
                    return m.group(1)
                return f"I had a processing issue, {self.M}. Please try again."

            except Exception as e:
                err = str(e)
                if ("429" in err or "rate" in err.lower() or "quota" in err.lower()):
                    log.warning(f"Groq key #{self._groq_idx+1} quota exhausted.")
                    rotated = self._rotate_groq(active_key)
                    if rotated:
                        self._say(f"Switching to backup key, {self.M}.")
                        continue
                    else:
                        self._say(f"All Groq keys exhausted, {self.M}. "
                                  "Waiting one minute for quota to reset.")
                        time.sleep(60)
                        self._reset_exhausted()
                        self._say(f"Ready again, {self.M}. Please ask again.")
                        return ""
                else:
                    log.error(f"Groq error: {e}")
                    return f"I encountered an issue, {self.M}. Please try again."

        return f"All Groq keys are exhausted, {self.M}. Please try again shortly."

        # ── Gemini backend (new google-genai SDK, with auto key rotation) ─ #
    def _process_gemini(self, user_text: str) -> str:
        import time
        raw = ""

        keys_tried = 0
        total_keys = len(self._gemini_keys)

        while keys_tried < total_keys:
            client, active_key = self._active_gemini_model()
            keys_tried += 1

            try:
                # Build full prompt with system instruction + history
                system_txt = SYSTEM_PROMPT.replace("{master}", self.M)
                history_txt = ""
                for msg in self._history[-10:]:
                    role = "User" if msg["role"] == "user" else "JARVIS"
                    history_txt += f"{role}: {msg['content']}\n"

                full_prompt = (
                    f"{system_txt}\n\n"
                    f"Conversation so far:\n{history_txt}\n"
                    f"User: {user_text}\n"
                    f"JARVIS (respond with JSON only):"
                )

                # Up to 2 quick retries on same key
                for attempt in range(2):
                    try:
                        resp = client.models.generate_content(
                            model=self._gemini_model_name,
                            contents=full_prompt,
                            config={"temperature": 0.4, "max_output_tokens": 400},
                        )
                        raw = resp.text.strip()
                        break
                    except Exception as e:
                        err = str(e)
                        if ("429" in err or "quota" in err.lower()
                                or "rate" in err.lower() or "exhausted" in err.lower()):
                            if attempt == 0:
                                log.warning(f"Key #{self._gemini_idx+1} rate limited. Retry in 3s...")
                                time.sleep(3)
                            else:
                                raise
                        else:
                            raise

                # Clean response
                raw = re.sub(r"```json", "", raw)
                raw = re.sub(r"```",     "", raw).strip()
                # Extract JSON if surrounded by text
                json_match = re.search(r"\{.*\}", raw, re.DOTALL)
                if json_match:
                    raw = json_match.group(0)

                log.debug(f"Gemini key #{self._gemini_idx+1}: {raw[:80]}")

                plan   = json.loads(raw)
                spoken = self._execute(plan)

                self._history.append({"role": "user",      "content": user_text})
                self._history.append({"role": "assistant",  "content": raw})
                if len(self._history) > 14:
                    self._history = self._history[-14:]

                return spoken

            except json.JSONDecodeError:
                m = re.search(r'"speak"\s*:\s*"([^"]+)"', raw)
                if m:
                    return m.group(1)
                return f"I had a processing issue, {self.M}. Please try again."

            except Exception as e:
                err = str(e)
                if ("429" in err or "quota" in err.lower()
                        or "rate" in err.lower() or "exhausted" in err.lower()):
                    log.warning(f"Gemini key #{self._gemini_idx+1} quota exhausted.")
                    rotated = self._rotate_gemini(active_key)
                    if rotated:
                        log.info(f"Switching to Gemini key #{self._gemini_idx+1}...")
                        self._say(f"Switching to backup key, {self.M}. One moment.")
                        continue
                    else:
                        log.warning("All Gemini keys exhausted. Waiting 60s...")
                        self._say(
                            f"All API keys have reached their quota, {self.M}. "
                            "I am waiting one minute for the limits to reset. "
                            "Please hold on.")
                        time.sleep(60)
                        self._reset_exhausted()
                        self._say(f"Quota reset, {self.M}. Please ask again.")
                        return ""
                else:
                    log.error(f"Gemini error: {e}")
                    return f"I encountered an issue, {self.M}. Please try again."

        return f"All Gemini keys are exhausted, {self.M}. Please try again shortly."

    # ── OpenAI backend (with auto key rotation) ──────────────────── #
    def _process_openai(self, user_text: str) -> str:
        import time

        keys_tried = 0
        total_keys = len(self._openai_keys)

        while keys_tried < total_keys:
            client, active_key = self._active_openai_client()
            keys_tried += 1

            try:
                self._history.append({"role": "user", "content": user_text})
                if len(self._history) > 12:
                    self._history = self._history[-12:]

                system   = SYSTEM_PROMPT.replace("{master}", self.M)
                messages = [{"role": "system", "content": system}] + self._history

                response = client.chat.completions.create(
                    model=self.cfg.openai_model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.4,
                    response_format={"type": "json_object"},
                )

                raw    = response.choices[0].message.content.strip()
                log.debug(f"OpenAI key #{self._openai_idx+1}: {raw[:80]}")
                plan   = json.loads(raw)
                spoken = self._execute(plan)
                self._history.append({"role": "assistant", "content": raw})
                return spoken

            except json.JSONDecodeError as e:
                log.error(f"OpenAI JSON parse: {e}")
                return f"I had trouble processing that, {self.M}. Please try again."

            except Exception as e:
                err = str(e)
                if ("429" in err or "quota" in err.lower()
                        or "rate_limit" in err.lower() or "insufficient_quota" in err.lower()):
                    log.warning(f"OpenAI key #{self._openai_idx+1} quota exhausted.")
                    rotated = self._rotate_openai(active_key)
                    if rotated:
                        log.info(f"Switching to OpenAI key #{self._openai_idx+1}...")
                        self._say(f"Switching to backup API key, {self.M}.")
                        continue
                    else:
                        self._say(f"All API keys exhausted, {self.M}. Waiting one minute.")
                        time.sleep(60)
                        self._reset_exhausted()
                        self._say(f"Quota reset, {self.M}. Please try again.")
                        return ""
                else:
                    log.error(f"OpenAI error: {e}")
                    return f"I encountered an issue, {self.M}. Please try again."

        return f"All OpenAI keys are exhausted, {self.M}. Please try again shortly."

    # ── Execute action from AI plan ────────────────────────────────── #
    def _execute(self, plan: dict) -> str:
        action = plan.get("action", "speak_only")
        params = plan.get("params", {})
        speak  = plan.get("speak", "Done.")

        # Fill master name placeholder
        speak = speak.replace("{master}", self.M)

        log.debug(f"Executing action='{action}' params={params}")

        try:
            if action == "open_website":
                return self._do_open_website(params, speak)

            elif action == "open_app":
                return self._do_open_app(params, speak)

            elif action == "system_info":
                return self._do_system_info(params, speak)

            elif action == "web_search":
                return self._do_web_search(params, speak)

            elif action == "time_date":
                return self._do_time_date(params, speak)

            elif action == "system_control":
                return self._do_system_control(params, speak)

            elif action == "volume":
                return self._do_volume(params, speak)

            elif action == "media_control":
                return self._do_media(params, speak)

            elif action == "create_note":
                return self._do_note(params, speak)

            elif action == "screenshot":
                return self._do_screenshot(speak)

            elif action == "open_folder":
                return self._do_open_folder(params, speak)

            elif action == "type_text":
                return self._do_type_text(params, speak)

            elif action == "speak_only":
                return speak

            else:
                log.warning(f"Unknown action: {action}")
                return speak

        except Exception as e:
            log.error(f"Action '{action}' failed: {e}")
            return f"I ran into a problem executing that, {self.M}."

    # ══════════════════════════════════════════════════════════════════ #
    #  Action Handlers
    # ══════════════════════════════════════════════════════════════════ #

    def _do_open_website(self, params: dict, speak: str) -> str:
        url = params.get("url", "")
        if not url:
            return f"I did not receive a URL to open, {self.M}."
        # Ensure proper URL format
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        log.info(f"Opened URL: {url}")
        return speak

    def _do_open_app(self, params: dict, speak: str) -> str:
        app = params.get("app", "").lower().strip()

        # Map app names to executable commands
        APP_MAP = {
            "chrome":       ["chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
            "firefox":      ["firefox", r"C:\Program Files\Mozilla Firefox\firefox.exe"],
            "notepad":      ["notepad.exe"],
            "vscode":       ["code",
                             os.path.join(os.environ.get("LOCALAPPDATA",""),
                                          r"Programs\Microsoft VS Code\Code.exe")],
            "calculator":   ["calc.exe"],
            "explorer":     ["explorer.exe"],
            "paint":        ["mspaint.exe"],
            "taskmgr":      ["taskmgr.exe"],
            "spotify":      [os.path.join(os.environ.get("APPDATA",""),
                                          r"Spotify\Spotify.exe"), "spotify.exe"],
            "word":         [r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE", "winword.exe"],
            "excel":        [r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE", "excel.exe"],
            "discord":      [os.path.join(os.environ.get("LOCALAPPDATA",""),
                                          r"Discord\Update.exe"), "discord.exe"],
            "steam":        [r"C:\Program Files (x86)\Steam\Steam.exe", "steam.exe"],
            "vlc":          [r"C:\Program Files\VideoLAN\VLC\vlc.exe", "vlc.exe"],
            "powershell":   ["powershell.exe"],
            "cmd":          ["cmd.exe"],
            "settings":     ["ms-settings:"],
            "control":      ["control"],
        }

        cmds = APP_MAP.get(app, [app])

        for cmd in cmds:
            try:
                subprocess.Popen(cmd, shell=True)
                log.info(f"Launched app: {cmd}")
                return speak
            except Exception:
                continue

        return f"I was unable to open {app}, {self.M}. It may not be installed."

    def _do_system_info(self, params: dict, speak: str) -> str:
        info_type = params.get("type", "").lower()

        try:
            import psutil

            if info_type == "battery":
                b = psutil.sensors_battery()
                if b:
                    pct    = f"{b.percent:.0f}"
                    status = "charging" if b.power_plugged else "discharging"
                    speak  = speak.replace("{battery_percent}", pct)
                    speak  = speak.replace("{battery_status}", status)
                    # If AI didn't use placeholders, build default
                    if "{" not in speak:
                        return speak
                    return f"Battery is at {pct} percent and currently {status}."
                return f"This device does not report battery information, {self.M}."

            elif info_type == "cpu":
                pct   = psutil.cpu_percent(interval=1)
                cores = psutil.cpu_count(logical=True)
                speak = speak.replace("{cpu_percent}", f"{pct:.0f}")
                speak = speak.replace("{cpu_cores}", str(cores))
                if "{" not in speak:
                    return speak
                return f"CPU usage is {pct:.0f} percent across {cores} logical cores."

            elif info_type == "ram":
                m     = psutil.virtual_memory()
                speak = speak.replace("{ram_percent}", f"{m.percent:.0f}")
                speak = speak.replace("{ram_available}", f"{m.available/1e9:.1f}")
                speak = speak.replace("{ram_total}", f"{m.total/1e9:.1f}")
                if "{" not in speak:
                    return speak
                return (f"RAM is {m.percent:.0f} percent used. "
                        f"{m.available/1e9:.1f} gigabytes available of {m.total/1e9:.1f} total.")

            elif info_type == "disk":
                d     = psutil.disk_usage("C:\\")
                speak = speak.replace("{disk_percent}", f"{d.percent:.0f}")
                speak = speak.replace("{disk_free}", f"{d.free/1e9:.1f}")
                speak = speak.replace("{disk_total}", f"{d.total/1e9:.1f}")
                if "{" not in speak:
                    return speak
                return (f"C drive is {d.percent:.0f} percent full. "
                        f"{d.free/1e9:.1f} gigabytes free of {d.total/1e9:.1f} total.")

            elif info_type == "ip":
                ip = socket.gethostbyname(socket.gethostname())
                speak = speak.replace("{ip}", ip)
                if "{" not in speak:
                    return speak
                return f"Your local IP address is {ip}."

            elif info_type == "uptime":
                import time
                s    = time.time() - psutil.boot_time()
                h, r = divmod(int(s), 3600)
                m, _ = divmod(r, 60)
                speak = speak.replace("{uptime_hours}", str(h))
                speak = speak.replace("{uptime_minutes}", str(m))
                if "{" not in speak:
                    return speak
                return f"System has been running for {h} hours and {m} minutes."

            elif info_type == "username":
                user = os.environ.get("USERNAME", "unknown")
                host = socket.gethostname()
                speak = speak.replace("{username}", user).replace("{hostname}", host)
                if "{" not in speak:
                    return speak
                return f"You are logged in as {user} on {host}."

        except ImportError:
            return "Please install psutil for system information. Run: pip install psutil"

        return speak

    def _do_web_search(self, params: dict, speak: str) -> str:
        query = params.get("query", "").strip()
        if not query:
            return f"What would you like me to search for, {self.M}?"
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        log.info(f"Searched: {query}")
        return speak

    def _do_time_date(self, params: dict, speak: str) -> str:
        now      = datetime.datetime.now()
        now_time = now.strftime("%I:%M %p")
        now_date = now.strftime("%A, %B %d, %Y")
        now_day  = now.strftime("%A")
        now_year = str(now.year)

        speak = speak.replace("{time}", now_time)
        speak = speak.replace("{date}", now_date)
        speak = speak.replace("{day}",  now_day)
        speak = speak.replace("{year}", now_year)

        # If AI didn't use placeholders, build a sensible default
        if "{" in speak:
            dtype = params.get("type", "time")
            if dtype == "time":
                return f"The current time is {now_time}, {self.M}."
            elif dtype == "date":
                return f"Today is {now_date}."
            elif dtype == "day":
                return f"Today is {now_day}."
            elif dtype == "year":
                return f"The current year is {now_year}."
            else:
                return f"It is {now_time} on {now_date}."
        return speak

    def _do_system_control(self, params: dict, speak: str) -> str:
        action = params.get("action", "").lower()
        if action == "shutdown":
            os.system("shutdown /s /t 15")
        elif action == "restart":
            os.system("shutdown /r /t 15")
        elif action == "sleep":
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif action == "lock":
            os.system("rundll32.exe user32.dll,LockWorkStation")
        elif action == "logoff":
            os.system("shutdown /l")
        elif action == "cancel_shutdown":
            os.system("shutdown /a")
        return speak

    def _do_volume(self, params: dict, speak: str) -> str:
        direction = params.get("direction", "up").lower()
        # Try pycaw first (precise control)
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            dev   = AudioUtilities.GetSpeakers()
            iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            v     = cast(iface, POINTER(IAudioEndpointVolume))
            cur   = v.GetMasterVolumeLevelScalar()
            if direction == "up":
                v.SetMasterVolumeLevelScalar(min(1.0, cur + 0.1), None)
            elif direction == "down":
                v.SetMasterVolumeLevelScalar(max(0.0, cur - 0.1), None)
            elif direction in ("mute", "unmute"):
                v.SetMute(not v.GetMute(), None)
            return speak
        except Exception:
            pass
        # Fallback: keyboard simulation
        try:
            import pyautogui
            key_map = {"up": "volumeup", "down": "volumedown",
                       "mute": "volumemute", "unmute": "volumemute"}
            pyautogui.press(key_map.get(direction, "volumeup"))
        except Exception:
            pass
        return speak

    def _do_media(self, params: dict, speak: str) -> str:
        action = params.get("action", "play_pause").lower()
        try:
            import pyautogui
            key_map = {
                "play_pause": "playpause",
                "next":       "nexttrack",
                "previous":   "prevtrack",
            }
            pyautogui.press(key_map.get(action, "playpause"))
        except Exception as e:
            log.error(f"Media control: {e}")
        return speak

    def _do_note(self, params: dict, speak: str) -> str:
        content = params.get("content", "").strip()

        # If AI didn't extract content, ask user
        if not content:
            return "__AWAIT_NOTE__"

        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.cfg.notes_folder) / f"note_{ts}.txt"
        path.write_text(
            f"JARVIS Note — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
            + "-" * 40 + "\n" + content, encoding="utf-8")
        log.info(f"Note saved: {path}")
        return speak

    def save_note_content(self, content: str) -> str:
        """Called after listening for note content."""
        if not content:
            return f"I did not catch anything to write, {self.M}."
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.cfg.notes_folder) / f"note_{ts}.txt"
        path.write_text(
            f"JARVIS Note — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
            + "-" * 40 + "\n" + content, encoding="utf-8")
        return f"Note saved to your Desktop in the JARVIS Notes folder."

    def _do_type_text(self, params: dict, speak: str) -> str:
        """
        Types text into whatever app is currently active/focused.
        Uses pyautogui.typewrite for ASCII and pyperclip+paste for Unicode.
        params:
            text  — the text to type
            mode  — "type" (just type) | "enter" (type then press Enter)
        """
        text_to_type = params.get("text", "").strip()
        mode         = params.get("mode", "type").lower()

        if not text_to_type:
            return f"I did not receive any text to type, {self.M}."

        try:
            import pyautogui
            import time

            # Small delay so JARVIS voice finishes before typing starts
            time.sleep(0.6)

            # Try clipboard method first (supports all Unicode, much faster)
            try:
                import pyperclip
                pyperclip.copy(text_to_type)
                pyautogui.hotkey("ctrl", "v")
            except ImportError:
                # Fallback: typewrite (ASCII only, slower but no dependency)
                pyautogui.typewrite(text_to_type, interval=0.03)

            if mode == "enter":
                time.sleep(0.1)
                pyautogui.press("enter")
                return f"Typed and sent: {text_to_type}"

            return f"Done. I have typed: {text_to_type}"

        except ImportError:
            return "Please install pyautogui: pip install pyautogui"
        except Exception as e:
            log.error(f"Type text error: {e}")
            return f"I could not type that, {self.M}. Please try again."

    def _do_screenshot(self, speak: str) -> str:
        try:
            import pyautogui
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(self.cfg.screenshots_folder) / f"screenshot_{ts}.png"
            pyautogui.screenshot(str(path))
            log.info(f"Screenshot: {path}")
        except ImportError:
            return "Please install pyautogui for screenshots."
        except Exception as e:
            log.error(f"Screenshot: {e}")
            return f"Screenshot failed, {self.M}."
        return speak

    def _do_open_folder(self, params: dict, speak: str) -> str:
        name = params.get("name", "desktop").lower()
        folder_map = {
            "desktop":   Path.home() / "Desktop",
            "downloads": Path.home() / "Downloads",
            "documents": Path.home() / "Documents",
            "pictures":  Path.home() / "Pictures",
            "music":     Path.home() / "Music",
            "videos":    Path.home() / "Videos",
        }
        folder = folder_map.get(name, Path.home() / "Desktop")
        try:
            os.startfile(str(folder))
        except Exception as e:
            log.error(f"Open folder: {e}")
        return speak

    # ── No-AI fallback (basic built-in responses) ─────────────────── #
    def _no_ai_fallback(self, text: str) -> str:
        text = text.lower()

        # Time / Date (always useful even without AI)
        if "time" in text:
            return f"The current time is {datetime.datetime.now():%I:%M %p}, {self.M}."
        if "date" in text or "today" in text:
            return f"Today is {datetime.datetime.now():%A, %B %d, %Y}."

        # Basic open
        if text.startswith("open "):
            target = text[5:].strip()
            try:
                webbrowser.open(f"https://www.{target.replace(' ','')}.com")
                return f"Attempting to open {target}, {self.M}."
            except Exception:
                pass

        return (f"AI is not configured, {self.M}. "
                "Please set the OPENAI_API_KEY environment variable to enable full capabilities. "
                "I can still tell you the time, date, and open websites by name.")