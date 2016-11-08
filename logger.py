"""Module that creates and configures logger from python logging module."""
import logging

from config import LOGGING_LEVEL

LOGGING_LEVELS = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}

def init_logger():
    """Creates and configures logger.
    Returns:
        logging.Logger: Logger object.
    """
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LOGGING_LEVELS[LOGGING_LEVEL])
    return logger

LOGGER = init_logger()