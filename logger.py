import logging, datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class _CleanConsole(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return (record.levelno >= logging.WARNING or
                msg.startswith("JARVIS:") or
                msg.startswith("Heard:") or
                "starting" in msg.lower() or
                "online" in msg.lower() or
                "calibrat" in msg.lower())


def setup_logger(debug=False):
    logger = logging.getLogger("jarvis")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    fh  = logging.FileHandler(
        LOG_DIR / f"jarvis_{datetime.date.today():%Y%m%d}.log",
        encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)
    if not debug:
        ch.addFilter(_CleanConsole())

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
