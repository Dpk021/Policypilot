from __future__ import annotations


def friendly_service_error(error: Exception) -> str:
    """Turn common provider failures into guidance without exposing configuration values."""
    name = type(error).__name__
    message = str(error).lower()
    if name in {"AuthenticationError", "UnauthorizedException"} or "invalid api key" in message:
        return "Authentication failed. Check that each API key is complete, active, and has no quotes or spaces."
    if name in {"NotFoundError", "PineconeApiException"} and ("index" in message or "not found" in message):
        return "The Pinecone index was not found. Check PINECONE_INDEX_NAME in .env."
    if name == "RateLimitError" or "credit_balance_exhausted" in message:
        return "The OpenAI request was rate-limited or has no remaining credit. Check API billing, then try again."
    if name in {"APIConnectionError", "ConnectError"} or "connection error" in message:
        return "Could not reach OpenAI or Pinecone. Check your internet connection, VPN, proxy, or firewall."
    return "The request failed. Open the details below to see the provider's safe error message."
