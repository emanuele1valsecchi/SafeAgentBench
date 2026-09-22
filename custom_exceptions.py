from ai2thor.controller import Controller
import re

class CustomException(Exception):
    """Base exception to handle standard string messages."""
    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return f"{self.message}"

class BadActionFormat(CustomException):
    pass

class InteractionException(CustomException):
    pass

class HoldingObjectsException(BaseException):
    pass

class ObjectException(BaseException):
    pass

class ReceptacleException(BaseException):
    pass

class AIRequestException(BaseException):
    pass

class MetamorphicRelationException(BaseException):
    pass

class Ai2THORExecuterException(BaseException):
    pass

class Ai2THORException(Exception):
    def __init__(self, controller : Controller):
        super().__init__()
        # ^(?P<exception>[^:]+)       -> Captures from the start until the first colon
        # :\s*(?P<message>.*?)        -> Captures the main message lazily
        # (?:\.\.?\s*trace:|\s*trace:) -> Handles the "trace:" delimiter (and the double dots AI2-THOR sometimes outputs)
        # \s*(?P<trace>.*)            -> Captures the rest of the multiline string as the stack trace
        pattern = re.compile(
            r"^(?P<exception>[^:]+):\s*(?P<message>.*?)(?:\.\.?\s*trace:|\s*trace:)\s*(?P<trace>.*)", 
            re.DOTALL | re.IGNORECASE
        )

        match = pattern.search(controller.last_event.metadata['errorMessage'])

        self.exception = match.group("exception").strip() if match else "Ai2THORException"
        self.message = match.group("message").strip() if match else controller.last_event.metadata['errorMessage']
        self.trace = match.group("trace").strip() if match else ""
    
    def __str__(self):
        return f"\n{self.exception}: {self.message}\n {self.trace}\n"

    def is_collision(self):
        return self.exception == 'InvalidOperationException' and self.message.lower().startswith("collided")