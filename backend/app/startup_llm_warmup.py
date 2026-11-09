"""Prepare local reply-client resources before the first voice turn.

Constructing an SDK client loads Windows TLS state. This does not contact
the provider, validate a credential, or pre-generate an answer.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


async def warm_reply_client(timeout_seconds: float = 3.0) -> None:
    from config import (TUTOR_RESPONSE_API_KEY, TUTOR_RESPONSE_BASE_URL,
                        TUTOR_RESPONSE_MODEL)
    from llm_client import _resolve_route, robust_llm_client

    url, key = _resolve_route(
        TUTOR_RESPONSE_MODEL, TUTOR_RESPONSE_BASE_URL,
        TUTOR_RESPONSE_API_KEY, robust_llm_client.doubao_key,
    )
    if not key or not url:
        return
    try:
        # The existing async initializer shares one off-loop construction task
        # with concurrent turns. A startup timeout must not cancel that task.
        await asyncio.wait_for(
            robust_llm_client._get_client_async(key, url, 30),
            timeout=timeout_seconds,
        )
    except Exception as exc:
        # Startup remains available if local SDK/TLS setup fails. Do not log
        # exception text, endpoints or credentials here; the normal turn path
        # retains its existing retry/failover and metering behavior.
        logger.warning("Reply client preparation incomplete: %s", type(exc).__name__)
