import logging

class Logger:
    def __init__(self):
        logging.basicConfig(filename="blockchain.log", level=logging.INFO, format="%(asctime)s - %(message)s")
        self.logger = logging.getLogger()

    def log(self, message):
        print(message)  # Optional: Print to console
        self.logger.info(message)