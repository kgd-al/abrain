from typing import Optional

from rich.logging import RichHandler

import logging
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)
MAYBE_DEBUG = logging.DEBUG + 5

logging.addLevelName(MAYBE_DEBUG, 'MAYBE_DEBUG')


def get_next_tmp_data_root():
    root = Path("tmp")
    root.mkdir(exist_ok=True)

    next_id = max([int(str(path)
                       .split("/")[1]
                       .split("-")[0][3:]) for path in root.glob("run*/")] + [-1]) + 1
    return root.joinpath(f"run{next_id}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")


def setup_logging(folder: Path, header: Optional[str] = None):
    log = folder.joinpath("log")
    log.unlink(missing_ok=True)

    file_handler = logging.FileHandler(log)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s"))

    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(
        level=logging.WARNING,
        handlers=[file_handler, rich_handler],
    )

    logger.setLevel(logging.INFO)
    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(logging.INFO)

    w = 40
    logger.info("="*w)
    logger.info(f"{'New log starts here.':^{w}}")
    if header is not None:
        logger.info(f"{header:^{w}}")
    logger.info("="*w)

    assert log.exists()

    return logger
