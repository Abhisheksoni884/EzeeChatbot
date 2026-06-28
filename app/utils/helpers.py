"""
Utility helpers for EzeeChatBot.
"""

import uuid


def generate_bot_id() -> str:
    """
    Generate a unique, URL-safe bot identifier.
    Format: 'bot_<8-char-hex>' — short enough to type, unique enough to avoid collisions.
    """
    short_id = uuid.uuid4().hex[:8]
    return f"bot_{short_id}"
