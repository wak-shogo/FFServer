# notifications.py
import requests
import logging
import datetime

# ！！！重要：下のURLをご自身のDiscord Webhook URLに書き換えてください！！！
WEBHOOK_URL = "https://discord.com/api/webhooks/1367401525156118651/TIJeXfG0jHSVI8F4lL8VyJx_EevrR2T8KE1aNSlOVp_wmAi0B57qwbCdW9StvtK_3Yhx"

def send_to_discord(message, title="MD Simulation Update", color=3447003):
    """
    Discord notifications have been disabled by user request.
    """
    pass