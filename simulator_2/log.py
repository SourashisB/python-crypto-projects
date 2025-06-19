import logging

def setup_logger():
    logger = logging.getLogger('stock_exchange')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('exchange.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger

logger = setup_logger()