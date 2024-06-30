import logging

logger = logging.getLogger(__name__)
logger.handlers.clear()

formatter = logging.Formatter("[%(levelname)s] %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def set_debug(DEBUG=False):
    if DEBUG is False:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
