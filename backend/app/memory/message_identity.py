"""Conversation identity is a lookup hint, never authorization for a memory fact."""

MAX_TURN_ID = 128


def normalized_turn_id(value) -> str:
    if (not isinstance(value, str) or len(value) > MAX_TURN_ID
            or value != value.strip() or any(ord(char) < 32 for char in value)):
        return ""
    return value


def assistant_turn_id(message: dict) -> str:
    return normalized_turn_id(message.get("turn_id")) if message.get("role") == "assistant" else ""


def assistant_identity(message: dict) -> dict:
    if message.get("role") != "assistant":
        return {}
    if message.get("turn_identity_ambiguous") is True:
        return {"turn_identity_ambiguous": True}
    turn_id = assistant_turn_id(message)
    return {"turn_id": turn_id} if turn_id else {}
