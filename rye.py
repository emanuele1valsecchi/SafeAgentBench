import reelay
import json
import re
from utils import save_in_log
from ai2_thor_functionalities import get_object_type
from ai2_thor_functionalities import get_object_type_from_id
from ai2_thor_functionalities import is_on
from ai2_thor_functionalities import is_toggleable
from ai2_thor_functionalities import is_breakable
from ai2_thor_functionalities import is_broken
from ai2_thor_functionalities import can_contain_liquid
from ai2_thor_functionalities import contains_liquid
from ai2_thor_functionalities import get_liquid_inside
from ai2_thor_functionalities import is_dirtable
from ai2_thor_functionalities import is_dirty
from ai2_thor_functionalities import is_cookable
from ai2_thor_functionalities import is_cooked
from ai2_thor_functionalities import is_sliceable
from ai2_thor_functionalities import is_sliced
from ai2_thor_functionalities import is_openable
from ai2_thor_functionalities import is_completely_open
from ai2_thor_functionalities import is_pickupable
from ai2_thor_functionalities import is_picked_up
from ai2_thor_functionalities import get_object_parent_receptacles

class RyeManager:
    __DEFAULT_FILE = "./dataset/sys_behavior.json"

    def __init__(self):
        self.store_data : list[dict[str, str]] = []
        self.read_data  : list[dict[str, str]] = []
        self.errors     : list[dict[str, str]] = []

    def __add_missing_encodings(self):
        first_stored_data = self.store_data[0]

        for data in self.store_data[1:]:
            for k, v in data.items():
                if k not in first_stored_data.keys():
                    first_stored_data[k] = not v if (v == True or v == False) else 0

    def save_to_json(self, file : str = __DEFAULT_FILE):
        self.__add_missing_encodings()
        with open(file, "w", encoding='utf-8') as f:
            json.dump(self.store_data, f, ensure_ascii=False, indent=4)

            formatted_data = json.dumps(self.store_data, indent=4)
            save_in_log(f"\n\nRYE STORES DATA:\n{formatted_data}\n\n")

    def read_from_json(self, file : str = __DEFAULT_FILE):
        with open(file, "r") as f:
            self.read_data = json.load(f)

    def analysis(self, rye_pattern : str, last_state_only = False):

        self.errors.clear()

        if not rye_pattern or len(rye_pattern) < 1:
            raise Exception("The action cannot be performed without a Reelay Expression to test")

        if not self.read_data:
            self.read_from_json()

        rye_pattern = re.sub(r'[HP]\[0:0\]', '', rye_pattern)

        monitor = reelay.discrete_timed_monitor(pattern = repr(rye_pattern)[1:-1], condense = False)

        for i, data in enumerate(self.read_data):
            r = monitor.update(data)

            if r['value'] is False:
                if not last_state_only:
                    self.errors.append(f"Error at {monitor.now()}:\nRYE: '{rye_pattern}'\nNot respected\n")
                elif i == (len(self.store_data) - 1):
                    self.errors.append(f"RYE: '{rye_pattern}'\nNot respected\n")

        return list(self.errors)

    def get_errors(self):
        return list(self.errors)

    def update_data(self):
        self.read_data.clear()

        self.read_from_json()

    def update_state(self, new_data : dict[str, ]):
        self.store_data.append(new_data)

    # === SCENE STATE ENCODING === #

    def encode_scene_state(self, step_command : str, event_metadata : dict[str, str]):
        if not event_metadata:
            raise Exception("No event metadata provided")

        objects_data = event_metadata['objects']
        performed_action = self.__encode_performed_step(step_command)

        # Group states by 'objectType_state'
        state_collections = {}

        for obj in objects_data:
            obj_states = (self.encode_turnon(obj) | self.encode_object_broken(obj) | 
                        self.encode_liquid_inside(obj) | self.encode_dirty(obj) | 
                        self.encode_cooked(obj) | self.encode_slice(obj) | 
                        self.encode_open(obj) | self.encode_pick(obj))
            
            for k, v in obj_states.items():
                if k not in state_collections:
                    state_collections[k] = []
                state_collections[k].append(v)

        environment_state = {}

        for k, v_list in state_collections.items():
            if isinstance(v_list[0], bool):
                # Set a property to True if at least one of the objects with the same type has it to True
                environment_state[k] = any(v_list)

                # Set that all the objects with the same type have that property to True
                environment_state[f"all_{k}"] = all(v_list)

                # Counts all the objects that have a particularly state to True
                environment_state[f"{k}_count"] = sum(v_list)
            else:
                environment_state[k] = v_list[0]
                
            obj_type = k.rsplit('_', 1)[0]
            environment_state[f"{obj_type}_total"] = len(v_list)

        environment_state = self.__clear_non_changing_args(environment_state)
        #input(f"\n\nPerformed action: {performed_action}\nEnvironment State: {environment_state}\n\n")
        self.update_state(performed_action | environment_state)

    def __clear_non_changing_args(self, environment_state : dict[str, str]):
        # Used because dictionary can't change size while looping on items
        cleaned_state = environment_state.copy()

        for k, v in environment_state.items():
            for data in reversed(self.store_data):
                if k in data.keys():
                    if v == data[k]:
                        cleaned_state.pop(k)
                        break
                    else:
                        break

        return cleaned_state

    # === ACTION PERFORMED === #
    def __encode_performed_step(self, step_command : str):

        prev_command = ""

        if self.store_data:
            prev_command = next(iter(self.store_data[-1])) if self.store_data[-1] else ""

        new_command = {step_command.lower().strip().replace(" ", "_") : True}

        if prev_command:
            return new_command | {prev_command : False}
        else:
            return new_command  

    # === TURN ON/OFF === #

    def __turnon_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_on"
    
    def __turnoff_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_off"

    def encode_turnon(self, object : dict[str, str]):
        if is_toggleable(object):
            return {self.__turnon_encoding(object) : is_on(object),
                    self.__turnoff_encoding(object) : not(is_on(object))}
        return {}

    # === BREAK OBJECT === #
    def __broken_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_broken"

    def encode_object_broken(self, object : dict[str, str]):
        if is_breakable(object):
            return {self.__broken_encoding(object) : is_broken(object)}

        return {}

    # === FILL/EMPTY LIQUID === # 
    def __fill_encoding(self, object : dict[str, str], liquid : str = "none"):
        object_type = get_object_type(object).lower()

        contained_liquid = get_liquid_inside(object)

        if contained_liquid:
            return object_type + "_filled_with_" + contained_liquid
        else:
            return object_type + "_filled_with_" + liquid

    def __empty_encoding(self, object : dict[str, str]):
        return "empty_" + get_object_type(object).lower()

    def get_ex_contained_liquid(self, object : dict[str, str]):
        if self.store_data:
            for record in reversed(self.store_data):
                for k, _ in record.items():
                    if self.__fill_encoding(object, liquid = "") in k:
                        return k.split("_")[-1]

        return None
        
    def encode_liquid_inside(self, object : dict[str, str]):
        if can_contain_liquid(object):
            encoding = {self.__empty_encoding(object) : not(contains_liquid(object))}

            if not contains_liquid(object):
                ex_liquid = self.get_ex_contained_liquid(object)

                if not ex_liquid:
                    return encoding

                return encoding | {self.__fill_encoding(object, liquid = ex_liquid) : False}

            return encoding | {self.__fill_encoding(object) : True}

        return {}

    # === DIRTY === #
    def __dirty_encoding(self, object: dict[str, str]):
        return get_object_type(object).lower() + "_dirty"

    def __clean_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_clean"

    def encode_dirty(self, object : dict[str, str]):
        if is_dirtable(object):
            return {self.__dirty_encoding(object) : is_dirty(object),
                    self.__clean_encoding(object) : is_dirty(object)}
        return {}
    
    # === COOKING === #
    def __cooked_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_cooked"

    def encode_cooked(self, object : dict[str, str]):
        if is_cookable(object):
            return {self.__cooked_encoding(object) : is_cooked(object)}

        return {}
    
    # === SLICING === #
    def __slice_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_sliced"

    def encode_slice(self, object : dict[str, str]):
        if is_sliceable(object):
            return {self.__slice_encoding(object) : is_sliced(object)}

        return {}

    # === OPEN/CLOSE === #
    def __open_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_open"

    def __close_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_close"

    def encode_open(self, object : dict[str, str]):
        if is_openable(object):
            return {self.__open_encoding(object) : is_completely_open(object),
                    self.__close_encoding(object) : not is_completely_open(object)}

        return {}

    # === PICK === #
    def __picked_encoding(self, object : dict[str, str]):
        return get_object_type(object).lower() + "_in_hand"

    def __receptacle_encoding(self, object : dict[str, str], receptacle = ""):
        object_type = get_object_type(object).lower()

        if receptacle == "none":
            receptacle = ""
        elif not receptacle:
            parent_receptacles = get_object_parent_receptacles(object)

            if parent_receptacles:
                receptacle = get_object_type_from_id(parent_receptacles[0]).lower()
        
        return  object_type + "_in_" + receptacle

    def get_ex_receptacle(self, object : dict[str, str]):
        receptacle_prefix = self.__receptacle_encoding(object, receptacle = "none")

        if self.store_data:
            for record in reversed(self.store_data):
                for k in record.keys():
                    k_parts = k.split("_")

                    ex_rec = k_parts[-1]
                    object_in = f"{k_parts[0]}_{k_parts[1]}_"

                    if receptacle_prefix == object_in:
                        if ex_rec != "hand" and ex_rec != "count" and ex_rec != "total":
                            return ex_rec

        return None
    
    def encode_pick(self, object : dict[str, str]):
        encoding = {}

        if is_pickupable(object):
            is_picked = is_picked_up(object)

            encoding =  {self.__picked_encoding(object) : is_picked}

            if is_picked:
                ex_receptacle = self.get_ex_receptacle(object)

                if ex_receptacle:
                    encoding =  encoding | {self.__receptacle_encoding(object, receptacle = ex_receptacle) : not is_picked}

            if get_object_parent_receptacles(object):
                encoding = encoding | {self.__receptacle_encoding(object) : not is_picked}

        return encoding
