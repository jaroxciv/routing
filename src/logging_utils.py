# src/logging_utils.py

import logging

def get_logger(module_name):
    logger = logging.getLogger(module_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger
