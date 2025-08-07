# notifications.py
import requests
import logging
import datetime

# ！！！重要：下のURLをご自身のDiscord Webhook URLに書き換えてください！！！
WEBHOOK_URL = "https://discord.com/api/webhooks/1367401525156118651/TIJeXfG0jHSVI8F4lL8VyJx_EevrR2T8KE1aNSlOVp_wmAi0B57qwbCdW9StvtK_3Yhx"

def send_to_discord(message, title="MD Simulation Update", color=3447003):
    """
    指定されたメッセージをDiscordに送信する関数。
    color: 16進数カラーコード (例: 青=3447003, 緑=3066993, 赤=15158332)
    """
    if "YOUR_WEBHOOK_URL_HERE" in WEBHOOK_URL:
        logging.warning("Discord Webhook URL has not been set in notifications.py. Skipping notification.")
        return

    try:
        timestamp = datetime.datetime.now().isoformat()
        embed = {
            "title": f"📢 {title}",
            "description": message,
            "color": color,
            "footer": {"text": f"Notification sent at {timestamp}"}
        }
        response = requests.post(WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
        response.raise_for_status()
        logging.info("Discord notification successful.")
    except Exception as e:
        logging.error(f"Failed to send Discord notification: {e}")