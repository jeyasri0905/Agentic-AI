"""
datetime_tool.py - Zero-Dependency Date & Time Tool for Agentic AI
Uses Python standard library datetime to return live date and time.
"""

from datetime import datetime, timezone
from typing import Any, Dict

DATETIME_TOOL_DECLARATION = {
  "name": "get_current_datetime",
  "description": "Gets the current live date, time, day of the week, and timezone. Use whenever the user asks for the current time, today's date, day of week, current month, or current year.",
  "parameters": {
    "type": "OBJECT",
    "properties": {
      "timezone": {
        "type": "STRING",
        "description": "Optional timezone identifier: 'local' for local system time, or 'UTC' for Coordinated Universal Time. Defaults to 'local'."
      }
    }
  }
}


def get_current_datetime(timezone: str = "local", **kwargs) -> Dict[str, Any]:
    """
    Returns the current date and time.
    """
    tz_str = (timezone or "local").strip().lower()

    if tz_str == "utc":
        now = datetime.now(timezone.utc)
        tz_label = "UTC"
    else:
        now = datetime.now().astimezone()
        tz_label = now.tzname() or "Local"

    return {
        "status": "success",
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "year": now.year,
        "month": now.strftime("%B"),
        "day": now.day,
        "iso_timestamp": now.isoformat(),
        "timezone": tz_label
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_current_datetime(), indent=2))
