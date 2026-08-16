agent_action = ("find", "pick", "put", "drop", "throw", "moveheldback", "moveheldleft", "moveheldright", "moveheldup", "movehelddown", "pour", "push", "pull", "open", "close", "break", "cook", "slice", "turnon", "turnoff", "dirty", "clean", "fillliquid", "emptyliquid")

liquid_available = ("coffee", "wine", "water")

# Constants for agent actions
FIND = agent_action[0]
PICK = agent_action[1]
PUT = agent_action[2]
DROP = agent_action[3]
THROW = agent_action[4]
MOVEHELDBACK = agent_action[5]
MOVEHELDLEFT = agent_action[6]
MOVEHELDRIGHT = agent_action[7]
MOVEHELDUP = agent_action[8]
MOVEHELDDOWN = agent_action[9]
POUR = agent_action[10]
PUSH = agent_action[11]
PULL = agent_action[12]
OPEN = agent_action[13]
CLOSE = agent_action[14]
BREAK = agent_action[15]
COOK = agent_action[16]
SLICE = agent_action[17]
TURNON = agent_action[18]
TURNOFF = agent_action[19]
DIRTY = agent_action[20]
CLEAN = agent_action[21]
FILLLIQUID = agent_action[22]
EMPTYLIQUID = agent_action[23]

def get_action_from_cmd( command : str ) -> str:
    return command.strip().split(" ", 1)[0].lower().strip()

def get_subjects_from_cmd( command : str ) -> tuple[str]:
    return [subject for subject in command.split()[1:]]

def get_available_liquids():
    return liquid_available

def is_action( action : str ) -> bool:
    return action in agent_action

def is_liquid( liquid : str ) -> bool:
    return liquid in liquid_available

def get_no_object_requested_actions() -> tuple[str]:
    return (DROP, THROW, MOVEHELDBACK, MOVEHELDLEFT, MOVEHELDRIGHT, MOVEHELDUP, MOVEHELDDOWN)

def get_one_object_requested_actions() -> tuple[str]:
    return (FIND, PICK, PUT, PUSH, OPEN, CLOSE, BREAK, COOK, SLICE, TURNON, TURNOFF, DIRTY, CLEAN, EMPTYLIQUID)

def get_two_objects_requested_actions() -> tuple[str]:
    return (FILLLIQUID)