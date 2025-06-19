import threading
import json
import time
from log import logger

class Persistence:
    def __init__(self, exchange, filename="trades.json"):
        self.exchange = exchange
        self.filename = filename
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._save_trades_periodically, daemon=True)
        self.thread.start()

    def _save_trades_periodically(self):
        while not self._stop_event.is_set():
            self.save_trades()
            time.sleep(60)

    def stop(self):
        self._stop_event.set()
        self.thread.join()

    def save_trades(self):
        try:
            trades = [trade.to_dict() for trade in self.exchange.trades]
            with open(self.filename, 'w') as f:
                json.dump(trades, f, indent=2)
            logger.info(f"Saved {len(trades)} trades to {self.filename}")
        except Exception as e:
            logger.error(f"Error saving trades: {e}")