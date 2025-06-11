import os
import json
import requests

BITPAY_API_TOKEN = os.environ.get("BITPAY_API_TOKEN")  # Store securely in Lambda env vars
BITPAY_API_URL = "https://bitpay.com"

def lambda_handler(event, context):
    action = event.get("action")
    if action == "create_invoice":
        return create_invoice(event)
    elif action == "create_payout":
        return create_payout(event)
    else:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid action"})
        }

def create_invoice(event):
    url = f"{BITPAY_API_URL}/invoices"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BITPAY_API_TOKEN}"
    }
    payload = {
        "price": event.get("amount"),
        "currency": event.get("currency", "USD"),
        "buyerEmail": event.get("buyer_email", ""),
        "itemDesc": event.get("description", ""),
    }
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        return {
            "statusCode": 200,
            "body": r.text
        }
    else:
        return {
            "statusCode": r.status_code,
            "body": r.text
        }

def create_payout(event):
    url = f"{BITPAY_API_URL}/payouts"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BITPAY_API_TOKEN}"
    }
    payload = {
        "amount": event.get("amount"),
        "currency": event.get("currency", "USD"),
        "instructions": [
            {
                "label": event.get("label", "Payout"),
                "address": event.get("address"),
                "amount": event.get("amount"),
            }
        ]
    }
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code == 200:
        return {
            "statusCode": 200,
            "body": r.text
        }
    else:
        return {
            "statusCode": r.status_code,
            "body": r.text
        }