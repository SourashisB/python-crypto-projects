from log import logger

class Admin:
    def __init__(self, exchange):
        self.exchange = exchange

    def list_trades(self):
        logger.info("Admin accessed trade list.")
        for trade in self.exchange.trades:
            print(f"{trade.timestamp}: {trade.quantity} {trade.stock} @ {trade.price} | "
                  f"Buyer: {trade.buyer_id}, Seller: {trade.seller_id}")