import logging
from pathlib import Path

LOG_FILE = Path(__file__).with_name("logs") / "app.log"


def setup_logging(level=logging.INFO):
    LOG_FILE.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        encoding="utf-8",
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
