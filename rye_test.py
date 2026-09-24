import reelay
from utils import print_separator
from rye import RyeManager
import re

def get_action(data : dict[str, bool]):
    for k, v in data.items():
        return k, v

print_separator(title = "MANUAL")

sys_behavior : list[dict[str, bool]] = [
    {"find_fridge" : True, 
     "fridge_open" : False, 
     "fridge_close" : True, 
     "egg_in_fridge" : True, 
     "open_fridge" : False, 
     "find_egg" : False, 
     "pick_egg" : False,
     "egg_in_hand" : False,
     "close_fridge" : False,
     "find_sink" : False,
     "put_sink" : False,
     "egg_in_sink" : False,
     "find_faucet" : False,
     "turnon_faucet" : False,
     "faucet_on" : False,
     "faucet_off" : True},

    {"open_fridge" : True, "find_fridge" : False, "fridge_open" : True, "fridge_close" : False},
    {"find_egg" : True, "open_fridge" : False},
    {"pick_egg" : True, "find_egg" : False, "egg_in_hand" : True, "egg_in_fridge" : False},
    {"find_fridge" : True, "pick_egg" : False},
    {"close_fridge" : True, "find_fridge" : False, "fridge_open" : False, "fridge_close" : True},
    {"find_sink" : True, "close_fridge" : False},
    {"put_sink" : True, "find_sink" : False, "egg_in_hand" : False, "egg_in_sink" : True},
    {"find_faucet" : True, "put_sink" : False},
    {"turnon_faucet" : True, "find_faucet" : False, "faucet_on" : True, "faucet_off" : False},

    #{"open_fridge" : True, "find_fridge" : False, "fridge_open" : True, "fridge_close" : False},
    #{"find_egg" : True, "open_fridge" : False},
    #{"pick_egg" : True, "find_egg" : False, "egg_in_hand" : True, "egg_in_fridge" : False},
    #{"find_sink" : True, "pick_egg" : False},
    #{"put_sink" : True, "find_sink" : False, "egg_in_hand" : False, "egg_in_sink" : True},
    #{"find_fridge" : True, "put_sink" : False},
    #{"close_fridge" : True, "find_fridge" : False, "fridge_open" : False, "fridge_close" : True},
    #{"find_faucet" : True, "put_sink" : False},
    #{"turnon_faucet" : True, "find_faucet" : False, "faucet_on" : True, "faucet_off" : False}
]

pattern = r"{close_fridge : true} -> H{pick_egg : false}"

#pattern = r"{close_fridge : true} && Y{egg_in_fridge : false}"

pattern = re.sub(r'[HP]\[0:0\]', '', pattern)

monitor = reelay.discrete_timed_monitor(
    pattern=pattern, condense=False)

for x in sys_behavior:
    r = monitor.update(x)
    action, action_state = get_action(x)
    if r["value"] is False:
        print(f"! ERROR {monitor.now()} -> {action}")
    else:
        print(f"+ OK {monitor.now()} -> {action}")

print_separator(title = "JSON")

json_behavior : list[dict[str, str]] = None

rye_manager = RyeManager()

rye_manager.read_from_json()
errors = rye_manager.analysis(rye_pattern = pattern)

if errors:
    for err in errors:
        print(err)
else:
    print("No error")