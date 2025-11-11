class BrowserError(Exception):
    """Base class for browser-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BrowserTimeOutError(BrowserError):
    """Raised when a browser operation times out."""

    def __init__(self, message: str):
        # ToDo: possibly add a screen shot from the driver's browser
        super().__init__("Browser operation timed out: " + message)


class BrowserUnknownModeError(BrowserError):
    """Raised when an unknown model is specified."""

    def __init__(self, message: str):
        super().__init__("This mode is not implemented in the backend: " + message)


class BrowserStayLoggedOutFailed(BrowserError):
    """Raised when the 'Stay logged out' link is not found."""

    def __init__(self, message: str):
        super().__init__("The 'Stay logged out' link was not found/clicked: " + message)


class LLMSessionError(Exception):
    """Base class for LLM session-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LogInError(LLMSessionError):
    """Raised when a browser operation times out."""

    def __init__(self, message: str):
        # ToDo: possibly add a screen shot from the driver's browser
        super().__init__("Could not login into DeepSeek: " + message)


class MessageNotSentError(LLMSessionError):
    """Raised when a message could not be sent."""

    def __init__(self, message: str):
        super().__init__("Message could not be sent: " + message)
