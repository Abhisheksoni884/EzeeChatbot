"""
Stats store — persists per-bot metrics to a JSON file.

Tracks:
  - total messages served
  - cumulative latency (to compute average)
  - estimated token cost in USD
  - count of unanswered questions (bot said it couldn't find the info)

Design: each bot's stats live under its bot_id key.
A lightweight JSON store is enough for this scope; swap for a DB if you scale.
"""

import json
import os
import threading
from typing import Dict, Any

_STATS_FILE = "data/stats.json"
_lock = threading.Lock()  # thread-safe file writes for concurrent requests


def _load() -> Dict[str, Any]:
    """Read the stats file; return empty dict if it doesn't exist yet."""
    if not os.path.exists(_STATS_FILE):
        return {}
    with open(_STATS_FILE, "r") as f:
        return json.load(f)


def _save(data: Dict[str, Any]) -> None:
    """Write stats back to disk atomically (under lock)."""
    os.makedirs(os.path.dirname(_STATS_FILE), exist_ok=True)
    with open(_STATS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _default_entry() -> Dict[str, Any]:
    return {
        "total_messages": 0,
        "total_latency_ms": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "unanswered_questions": 0,
    }


def record_message(
    bot_id: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    was_unanswered: bool,
) -> None:
    """
    Called after every /chat response to update metrics.

    Args:
        bot_id: the bot that handled the message
        latency_ms: end-to-end response time in milliseconds
        input_tokens: prompt token count (estimated)
        output_tokens: completion token count (estimated)
        was_unanswered: True when the bot returned the fallback message
    """
    with _lock:
        data = _load()
        if bot_id not in data:
            data[bot_id] = _default_entry()

        entry = data[bot_id]
        entry["total_messages"] += 1
        entry["total_latency_ms"] += latency_ms
        entry["total_input_tokens"] += input_tokens
        entry["total_output_tokens"] += output_tokens
        if was_unanswered:
            entry["unanswered_questions"] += 1

        _save(data)


def get_stats(bot_id: str) -> Dict[str, Any]:
    """Return raw stats dict for a bot_id, or None if not found."""
    with _lock:
        data = _load()
    return data.get(bot_id)


def bot_exists_in_stats(bot_id: str) -> bool:
    with _lock:
        data = _load()
    return bot_id in data


def init_bot_stats(bot_id: str) -> None:
    """Create a fresh stats entry when a new bot is uploaded."""
    with _lock:
        data = _load()
        if bot_id not in data:
            data[bot_id] = _default_entry()
        _save(data)
