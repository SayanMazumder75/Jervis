"""
core/face_auth.py  —  JARVIS Face Recognition Login
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features:
  • First run  → registers YOUR face (takes 5 photos automatically)
  • Every start → verifies face before JARVIS activates
  • Stranger detected → JARVIS denies access and speaks a warning
  • Face away   → optional auto-lock after N seconds
  • Works fully offline (no API needed — runs on device)

Dependencies:
  pip install opencv-python face-recognition numpy

face_recognition uses dlib under the hood.
On Windows you may also need:
  pip install cmake
  pip install dlib
"""

import os
import cv2
import numpy as np
import pickle
import time
import threading
import logging
from pathlib import Path

log = logging.getLogger("jarvis")

# ── Paths ──────────────────────────────────────────────────────────── #
BASE_DIR      = Path(__file__).parent.parent
FACE_DATA_DIR = BASE_DIR / "face_data"
FACE_DATA_DIR.mkdir(exist_ok=True)
ENCODINGS_FILE = FACE_DATA_DIR / "owner_encodings.pkl"
SAMPLES_DIR    = FACE_DATA_DIR / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────── #
REGISTER_SAMPLES   = 8      # photos taken during registration
MATCH_THRESHOLD    = 0.5    # lower = stricter (0.4 strict, 0.6 lenient)
VERIFY_ATTEMPTS    = 30     # frames to try before failing
AUTO_LOCK_SECONDS  = 0      # 0 = disabled, else lock after N seconds away


class FaceAuth:
    """
    Handles face registration and verification for JARVIS.
    All processing is local — no internet required.
    """

    def __init__(self, cfg, tts=None):
        self.cfg          = cfg
        self.M            = cfg.master_name
        self.tts          = tts
        self._owner_encs  = []       # list of face encodings for owner
        self._watching    = False    # True when background watcher is active
        self._last_seen   = time.time()
        self._lock_cb     = None     # callback when face goes away too long
        self._fr          = None     # face_recognition module (lazy import)
        self._available   = False    # set True if face_recognition installed

        self._load_lib()
        self._load_encodings()

    # ── Import check ──────────────────────────────────────────────── #
    def _load_lib(self):
        try:
            import face_recognition
            self._fr = face_recognition
            self._available = True
            log.info("face_recognition library loaded")
        except ImportError:
            log.warning(
                "face_recognition not installed.\n"
                "Run: pip install cmake dlib face-recognition\n"
                "Face auth will be SKIPPED.")

    @property
    def available(self):
        return self._available

    @property
    def is_registered(self):
        return len(self._owner_encs) > 0

    # ── Load saved encodings ──────────────────────────────────────── #
    def _load_encodings(self):
        if ENCODINGS_FILE.exists():
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    self._owner_encs = pickle.load(f)
                log.info(f"Loaded {len(self._owner_encs)} face encodings for owner")
            except Exception as e:
                log.error(f"Could not load face encodings: {e}")
                self._owner_encs = []

    # ── Save encodings ────────────────────────────────────────────── #
    def _save_encodings(self):
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(self._owner_encs, f)
        log.info(f"Saved {len(self._owner_encs)} face encodings")

    # ════════════════════════════════════════════════════════════════ #
    #  REGISTRATION
    # ════════════════════════════════════════════════════════════════ #
    def register(self) -> bool:
        """
        Opens webcam, captures REGISTER_SAMPLES photos of owner,
        computes face encodings, saves to disk.
        Returns True on success.
        """
        if not self._available:
            return False

        self._say(
            f"Starting face registration, {self.M}. "
            "Please look directly at the camera. "
            "I will capture your face automatically.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._say("I could not access the webcam. Please check your camera.")
            return False

        encodings_collected = []
        samples_taken       = 0
        frame_count         = 0

        # Show live preview window
        cv2.namedWindow("JARVIS — Face Registration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JARVIS — Face Registration", 640, 480)

        while samples_taken < REGISTER_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs  = self._fr.face_locations(rgb, model="hog")
            display = frame.copy()

            if locs:
                for (top, right, bottom, left) in locs:
                    cv2.rectangle(display, (left, top), (right, bottom),
                                  (0, 255, 100), 2)

                # Capture every 8th frame (avoid blurry/duplicate)
                if frame_count % 8 == 0:
                    encs = self._fr.face_encodings(rgb, locs)
                    if encs:
                        encodings_collected.append(encs[0])
                        samples_taken += 1

                        # Save sample image
                        img_path = SAMPLES_DIR / f"sample_{samples_taken}.jpg"
                        cv2.imwrite(str(img_path), frame)

                        cv2.putText(display,
                                    f"Captured {samples_taken}/{REGISTER_SAMPLES}",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 100), 2)
                        log.info(f"Face sample {samples_taken}/{REGISTER_SAMPLES} captured")
            else:
                cv2.putText(display, "No face detected — look at camera",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 100, 255), 2)

            # Progress bar
            progress = int((samples_taken / REGISTER_SAMPLES) * 620)
            cv2.rectangle(display, (10, 460), (10 + progress, 475),
                          (0, 255, 100), -1)
            cv2.rectangle(display, (10, 460), (630, 475),
                          (50, 50, 50), 2)

            cv2.imshow("JARVIS — Face Registration", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(encodings_collected) >= 3:
            self._owner_encs = encodings_collected
            self._save_encodings()
            self._say(
                f"Face registration complete, {self.M}. "
                f"I captured {len(encodings_collected)} samples. "
                "I will now recognise you every time.")
            return True
        else:
            self._say(
                f"Registration failed, {self.M}. "
                "Not enough face samples captured. Please try again in better lighting.")
            return False

    # ════════════════════════════════════════════════════════════════ #
    #  VERIFICATION
    # ════════════════════════════════════════════════════════════════ #
    def verify(self, silent: bool = False) -> bool:
        """
        Opens webcam, tries to match face against owner encodings.
        Returns True if owner face found, False otherwise.
        silent=True skips the intro speech (for background checks).
        """
        if not self._available:
            log.warning("face_recognition not available — skipping verification")
            return True   # allow through if library not installed

        if not self.is_registered:
            log.warning("No face registered — skipping verification")
            return True   # allow through if not set up yet

        if not silent:
            self._say(
                f"Initiating facial recognition, {self.M}. "
                "Please look at the camera.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._say("Webcam unavailable. Bypassing face check.")
            return True

        verified    = False
        attempts    = 0
        frame_count = 0

        # Show verification window
        cv2.namedWindow("JARVIS — Face Verification", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("JARVIS — Face Verification", 640, 480)

        while attempts < VERIFY_ATTEMPTS:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 3 != 0:   # check every 3rd frame for speed
                # Still show display
                cv2.imshow("JARVIS — Face Verification", frame)
                cv2.waitKey(1)
                continue

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = self._fr.face_locations(rgb, model="hog")
            display = frame.copy()

            if locs:
                encs = self._fr.face_encodings(rgb, locs)
                for enc, (top, right, bottom, left) in zip(encs, locs):
                    distances = self._fr.face_distance(self._owner_encs, enc)
                    min_dist  = min(distances) if len(distances) > 0 else 1.0

                    if min_dist <= MATCH_THRESHOLD:
                        # ── OWNER RECOGNISED ──────────────────────── #
                        cv2.rectangle(display,
                                      (left, top), (right, bottom),
                                      (0, 255, 100), 3)
                        cv2.putText(display,
                                    f"OWNER VERIFIED ({1-min_dist:.0%} match)",
                                    (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 100), 2)
                        cv2.imshow("JARVIS — Face Verification", display)
                        cv2.waitKey(500)
                        verified = True
                        break
                    else:
                        # ── STRANGER DETECTED ─────────────────────── #
                        cv2.rectangle(display,
                                      (left, top), (right, bottom),
                                      (0, 0, 255), 3)
                        cv2.putText(display,
                                    "UNKNOWN PERSON",
                                    (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 2)
                        attempts += 1
            else:
                cv2.putText(display, f"Scanning... ({attempts}/{VERIFY_ATTEMPTS})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 255), 2)
                attempts += 1

            cv2.imshow("JARVIS — Face Verification", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if verified:
                break

        cap.release()
        cv2.destroyAllWindows()

        return verified

    # ════════════════════════════════════════════════════════════════ #
    #  BACKGROUND WATCHER  (optional — auto-lock when face leaves)
    # ════════════════════════════════════════════════════════════════ #
    def start_watching(self, lock_callback=None):
        """
        Runs background thread watching webcam.
        If owner face not seen for AUTO_LOCK_SECONDS → calls lock_callback.
        """
        if not self._available or AUTO_LOCK_SECONDS <= 0:
            return

        self._lock_cb  = lock_callback
        self._watching = True
        t = threading.Thread(target=self._watch_loop,
                             daemon=True, name="FaceWatcher")
        t.start()
        log.info(f"Face watcher started — locks after {AUTO_LOCK_SECONDS}s away")

    def stop_watching(self):
        self._watching = False

    def _watch_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log.error("Face watcher: webcam unavailable")
            return

        frame_count = 0
        while self._watching:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            frame_count += 1
            if frame_count % 15 != 0:   # check every 15 frames
                continue

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = self._fr.face_locations(rgb, model="hog")

            if locs:
                encs = self._fr.face_encodings(rgb, locs)
                for enc in encs:
                    dists = self._fr.face_distance(self._owner_encs, enc)
                    if len(dists) > 0 and min(dists) <= MATCH_THRESHOLD:
                        self._last_seen = time.time()
                        break

            away_for = time.time() - self._last_seen
            if away_for > AUTO_LOCK_SECONDS:
                log.info(f"Owner away for {away_for:.0f}s — triggering lock")
                if self._lock_cb:
                    self._lock_cb()
                self._last_seen = time.time()   # reset so it doesn't spam

        cap.release()

    # ── Speak helper ──────────────────────────────────────────────── #
    def _say(self, text: str):
        if self.tts:
            self.tts.speak_sync(text)
        else:
            print(f"JARVIS: {text}")

    # ── Delete registration (re-register) ─────────────────────────── #
    def delete_registration(self):
        self._owner_encs = []
        if ENCODINGS_FILE.exists():
            ENCODINGS_FILE.unlink()
        import shutil
        if SAMPLES_DIR.exists():
            shutil.rmtree(SAMPLES_DIR)
            SAMPLES_DIR.mkdir()
        log.info("Face registration deleted")
        self._say(f"Face data cleared, {self.M}. Please register again.")