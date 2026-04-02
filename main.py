"""
J.A.R.V.I.S  —  AI Voice Assistant with Face Recognition Login
Run: python main.py

First run:  registers your face automatically
Every run:  verifies your face before JARVIS activates
"""
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.config    import Config
from core.logger    import setup_logger
from core.voice     import TTSEngine
from core.face_auth import FaceAuth
from core.brain     import JarvisBrain
from ui.hud         import JarvisHUD


def main():
    cfg    = Config()
    logger = setup_logger(cfg.debug_mode)
    logger.info("J.A.R.V.I.S starting...")

    # Boot TTS early so we can speak during face auth
    tts = TTSEngine(
        rate        = cfg.tts_rate,
        volume      = cfg.tts_volume,
        voice_index = cfg.tts_voice_index,
    )

    # ── Face Authentication ───────────────────────────────────────── #
    face = FaceAuth(cfg, tts)

    if face.available:
        if not face.is_registered:
            # First time — register owner face
            tts.speak_sync(
                f"Welcome, {cfg.master_name}. "
                "This is your first time running JARVIS with face recognition. "
                "I need to scan your face. Please look at the camera.")
            success = face.register()
            if not success:
                tts.speak_sync(
                    "Face registration failed. "
                    "Starting JARVIS without face protection this time.")
        else:
            # Every run — verify face before starting
            tts.speak_sync(
                "Initiating facial recognition. "
                "Please look at the camera.")
            verified = face.verify()

            if verified:
                tts.speak_sync(
                    f"Identity confirmed. Welcome back, {cfg.master_name}.")
            else:
                tts.speak_sync(
                    "Identity not recognised. "
                    "Access denied. "
                    "This incident has been logged.")
                logger.warning("FACE AUTH FAILED — unauthorised access attempt")
                _show_denied_screen()
                sys.exit(0)
    else:
        logger.info("face_recognition not installed — skipping face auth")
        logger.info("To enable: pip install cmake dlib face-recognition")

    # ── Launch JARVIS ──────────────────────────────────────────────── #
    brain = JarvisBrain(cfg, existing_tts=tts, face_auth=face)
    hud   = JarvisHUD(brain, face_auth=face)
    brain.start()
    hud.run()


def _show_denied_screen():
    import tkinter as tk, time
    try:
        root = tk.Tk()
        root.title("ACCESS DENIED")
        root.configure(bg="#080000")
        root.geometry("500x280")
        root.resizable(False, False)
        tk.Label(root, text="⛔  ACCESS DENIED",
                 font=("Courier New", 30, "bold"),
                 fg="#ff2222", bg="#080000").pack(expand=True, pady=40)
        tk.Label(root, text="Identity not recognised.\nThis attempt has been logged.",
                 font=("Courier New", 12),
                 fg="#881111", bg="#080000").pack()
        root.after(3000, root.destroy)
        root.mainloop()
    except Exception:
        time.sleep(3)


if __name__ == "__main__":
    main()