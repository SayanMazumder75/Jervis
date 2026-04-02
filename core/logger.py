import logging, datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class _JarvisFilter(logging.Filter):
    """Console shows only JARVIS speech + errors. Full log goes to file."""
    def filter(self, record):
        msg = record.getMessage()
        return (record.levelno >= logging.WARNING or
                msg.startswith("JARVIS:") or
                msg.startswith("Heard:") or
                "online" in msg.lower() or
                "starting" in msg.lower() or
                "calibrat" in msg.lower())


def setup_logger(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("jarvis")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    fmt      = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    full_fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S")

    # File — everything
    fh = logging.FileHandler(
        LOG_DIR / f"jarvis_{datetime.date.today():%Y%m%d}.log",
        encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(full_fmt)

    # Console — clean output only
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)
    if not debug:
        ch.addFilter(_JarvisFilter())

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger