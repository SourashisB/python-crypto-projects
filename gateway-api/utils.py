import random

def generate_random_order():
    order_type = random.choice(["buy", "sell"])
    account = f"account_{random.randint(1, 100)}"
    price = round(random.uniform(100, 500), 2)
    quantity = random.randint(1, 10)

    return {
        "type": order_type,
        "account": account,
        "price": price,
        "quantity": quantity,
    }