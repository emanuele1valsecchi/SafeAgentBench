agent_action = ("find", "pick", "put", "drop", "throw", "moveheldback", "moveheldleft", "moveheldright", "moveheldup", "movehelddown", "pour", "push", "pull", "open", "close", "break", "cook", "slice", "turnon", "turnoff", "dirty", "clean", "fillliquid", "emptyliquid")

# Constants for agent actions (use these in pattern matching and comparisons)
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

def cmd_translation( command: str ) -> tuple[str]:
    parts = command.strip().split(" ", 1)
    
    action = parts[0].lower().strip()
    target = parts[1].lower().strip() if len(parts) > 1 else None

    return action, target