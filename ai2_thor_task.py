from enum import Enum

class LIQUID(Enum):
    COFFEE = "coffee"
    WINE = "wine"
    WATER = "water"

    @classmethod
    def is_liquid(cls, liquid: str) -> bool:
        """Checks if a string matches any available liquid."""
        return any(liquid == item.value for item in cls)

    @classmethod
    def get_all(cls):
        return tuple(l.value for l in cls)

class ACTIONS(Enum):
    FIND = ("find", 1)
    PICK = ("pick", 1)
    PUT = ("put", 1)
    DROP = ("drop", 0)
    THROW = ("throw", 0)
    MOVEHELDBACK = ("moveheldback", 0)
    MOVEHELDLEFT = ("moveheldleft", 0)
    MOVEHELDRIGHT = ("moveheldright", 0)
    MOVEHELDUP = ("moveheldup", 0)
    MOVEHELDDOWN = ("movehelddown", 0)
    POUR = ("pour", 0)
    PUSH = ("push", 1)
    PULL = ("pull", 1)
    OPEN = ("open", 1)
    CLOSE = ("close", 1)
    BREAK = ("break", 1)
    COOK = ("cook", 1)
    SLICE = ("slice", 1)
    TURNON = ("turnon", 1)
    TURNOFF = ("turnoff", 1)
    DIRTY = ("dirty", 1)
    CLEAN = ("clean", 1)
    FILLLIQUID = ("fillliquid", 2)
    EMPTYLIQUID = ("emptyliquid", 1)

    def __new__(cls, 
            action: str, 
            objects_req: int):
        obj = object.__new__(cls)
        obj._value_ = action
        obj.objects_required = objects_req
        return obj

    @classmethod
    def is_action( cls, action : str ) -> bool:
        """Checks if a string matches any available action."""
        return any(action == item.value for item in cls)

    @classmethod
    def get_actions_with_objects_requested(cls, count : int) -> tuple[str]:
        return tuple(item.value for item in cls if item.objects_required == count)

    def __str__(self) -> str:
        return self._value_