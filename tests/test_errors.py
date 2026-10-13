from policypilot.errors import friendly_service_error


class AuthenticationError(Exception):
    pass


class APIConnectionError(Exception):
    pass


def test_authentication_error_is_actionable_without_configuration_values():
    message = friendly_service_error(AuthenticationError("incorrect API key"))
    assert "Authentication failed" in message
    assert "incorrect API key" not in message


def test_connection_error_is_actionable():
    message = friendly_service_error(APIConnectionError("Connection error"))
    assert "internet connection" in message
