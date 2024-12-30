import argparse
from order_book import OrderBook

def main():
    parser = argparse.ArgumentParser(description="Crypto Exchange API")
    parser.add_argument("type", choices=["buy", "sell"], help="Order type (buy or sell)")
    parser.add_argument("account", help="Account name")
    parser.add_argument("price", type=float, help="Price per unit")
    parser.add_argument("quantity", type=int, help="Quantity of units")

    args = parser.parse_args()

    order = {
        "type": args.type,
        "account": args.account,
        "price": args.price,
        "quantity": args.quantity,
    }

    order_book = OrderBook()

    if args.type == "buy":
        order_book.add_buy_order(order)
    else:
        order_book.add_sell_order(order)

    print(f"Order successfully added: {order}")

if __name__ == "__main__":
    main()