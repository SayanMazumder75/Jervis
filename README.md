# J.A.R.V.I.S — Voice-Only AI Assistant
> *"Just A Rather Very Intelligent System"*

A full Iron Man–style voice assistant for Windows.  
**JARVIS speaks every answer aloud — nothing is written on screen.**  
The HUD is purely visual: an animated arc-reactor ring that reacts to state.

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

> **PyAudio fix on Windows:**
> ```bash
> pip install pipwin && pipwin install pyaudio
> ```

---

## How it works

1. JARVIS starts → calibrates your microphone → **speaks** a greeting
2. Sits in **standby** (low-power listening)
3. You say **"Jarvis"** → HUD flashes, ring activates → JARVIS says *"Yes Sir, I'm listening"*
4. You say your command → JARVIS **speaks** the answer
5. Say **"goodbye"** or **"go to sleep"** → back to standby

---

## Voice Commands (50+)

| Category       | Say…                                                     |
|----------------|----------------------------------------------------------|
| **Wake**       | "Jarvis"                                                 |
| **Time/Date**  | "What time is it" · "What's the date" · "What year"     |
| **System**     | "Battery" · "CPU usage" · "RAM usage" · "Disk space" · "My IP" · "Uptime" |
| **Apps**       | "Open Chrome / Firefox / VS Code / Notepad / Spotify / Discord / Steam…" |
| **Web**        | "Open YouTube / Google / GitHub / Gmail / Netflix / Reddit…" |
| **Search**     | "Search for [anything]"                                  |
| **Volume**     | "Volume up / down / mute" · "Play pause" · "Next track" |
| **Files**      | "Create note" · "Screenshot" · "Open Desktop / Downloads / Documents" |
| **System Ops** | "Shutdown" · "Restart" · "Sleep" · "Lock" · "Sign out" · "Cancel shutdown" |
| **Fun**        | "Tell me a joke" · "Flip a coin" · "Roll a dice" · "Fun fact" |
| **Personality**| "Who are you" · "How are you" · "What can you do" · "Are you better than Siri" |
| **Sleep**      | "Goodbye" · "Go to sleep" · "Standby"                   |

---

## AI Mode (Optional)

For smart answers to anything not in the command list:

```bash
setx OPENAI_API_KEY "sk-your-key-here"
pip install openai
```

Restart terminal, then run `python main.py`. Any unknown question goes to GPT and JARVIS speaks the answer.

---

## Customise

Edit `config.json` (auto-created on first run):

```json
{
  "wake_word": "jarvis",
  "master_name": "Sir",
  "tts_rate": 165,
  "app_paths": {
    "chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
  }
}
```

Change `master_name` to your own name — JARVIS will address you by it.

---

## File Structure

```
jarvis_voice/
├── main.py              ← Entry point
├── config.json          ← Auto-generated settings
├── requirements.txt
├── core/
│   ├── brain.py         ← Wake loop, command dispatch, AI
│   ├── voice.py         ← TTSEngine + VoiceEngine (mic)
│   ├── commands.py      ← 50+ command handlers
│   ├── config.py        ← Settings loader
│   └── logger.py        ← File + console logging
└── ui/
    └── hud.py           ← Arc-reactor HUD (visual only)
```
