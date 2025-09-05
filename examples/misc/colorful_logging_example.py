import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests  # Example third-party lib

from cogents_core.utils.logging_config import get_logger, setup_logging


def main():
    setup_logging(level="DEBUG", log_file="logs/sample.log", enable_colors=True)

    logger = get_logger(__name__)

    logger.debug("This is a DEBUG message.")
    logger.info("This is an INFO message.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")

    try:
        requests.get("http://nonexistent.domain")
    except Exception:
        logger.exception("Caught an exception from requests:")


if __name__ == "__main__":
    main()
