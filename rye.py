import reelay
import json

class RyeManager:
    def __init__(self):
        self.store_data = []
        self.read_data = []
        self.errors = []

    def update_state(self, new_data : dict[str, any]):
        converted_data = {
            k.lower(): ("true" if v is True else ("false" if v is False else str(v))) 
            for k, v in new_data.items()
        }
        self.store_data.append(converted_data)

    def save_to_json(self):
        with open("./dataset/sys_behavior.json", "w", encoding='utf-8') as f:
            json.dump(self.store_data, f, ensure_ascii=False, indent=4)

    def read_from_json(self):
        with open("./dataset/sys_behavior.json", "r") as f:
            self.read_data = json.load(f)

    def analysis(self, rye_pattern : str):

        self.errors.clear()

        if not rye_pattern or len(rye_pattern) < 1:
            raise Exception("The action cannot be performed without a Reelay Expression to test")

        self.read_from_json()

        monitor = reelay.discrete_timed_monitor(pattern = repr(rye_pattern)[1:-1], condense = False)

        for x in self.read_data:
            r = monitor.update(x)

            if r['value'] is False:
                self.errors.append(f"Error at {monitor.now()}:\n RYE: '{rye_pattern}'\n was not respected\n")

    def get_errors(self):
        return self.errors
            

    def encode_empty_action(self):
        self.update_state({})

    def encode_pick(self, object_type : str, receptacle : str):
        self.update_state({object_type + "_in_hand" : True,
                           object_type + "_in_" + receptacle : False})

    def encode_put(self, object_type : str, receptacle : str):
        self.update_state({object_type + "_in_" + receptacle : True,
                           object_type + "_in_hand" : False})
        
    def encode_drop(self, object_type : str, receptacle : str):
        # Equal to encode_put, if needed can be modified for future releases
        self.update_state({object_type + "_in_" + receptacle : True,
                           object_type + "_in_hand" : False})

    def encode_throw(self, object_type : str, receptacle : str):
        self.update_state({object_type + "_in_" + receptacle : False,
                           object_type + "_in_hand" : False})

    def encode_moveheldback(self):
        self.update_state({})

    def encode_moveheldleft(self):
        self.update_state({})

    def encode_moveheldright(self):
        self.update_state({})

    def encode_moveheldup(self):
        self.update_state({})

    def encode_moveheldown(self):
        self.update_state({})

    def encode_pour(self, object_type : str, liquid : str):
        self.update_state({"pour_" + liquid : True, 
                           "empty_" + object_type : True,
                           object_type + "_filled_with_" + liquid : False})
        self.update_state({"pour_" + liquid : False})

    def encode_push(self, object_type : str):
        self.update_state({"push_" + object_type : True})
        self.update_state({"push_" + object_type : False})

    def encode_pull(self, object_type : str):
        self.update_state({"pull_" + object_type : True})
        self.update_state({"pull_" + object_type : False})

    def encode_open(self, object_type : str):
        self.update_state({object_type + "_open" : True,
                           object_type + "_close" : False})

    def encode_close(self, object_type : str):
        self.update_state({object_type + "_open" : False,
                           object_type + "_close" : True})

    def encode_break(self, object_type : str):
        self.update_state({object_type + "_broken" : True})

    def encode_cook(self, object_type : str):
        self.update_state({object_type + "_cooked" : True})

    def encode_slice(self, object_type : str):
        self.update_state({object_type + "_sliced" : True})

    def encode_turnon(self, object_type : str):
        self.update_state({object_type + "_on" : True,
                           object_type + "_off" : False})

    def encode_turnoff(self, object_type : str):
        self.update_state({object_type + "_on" : False,
                           object_type + "_off" : True})  

    def encode_dirty(self, object_type : str):
        self.update_state({object_type + "_dirty" : True,
                           object_type + "_cleaned" : False})

    def encode_clean(self, object_type : str):
        self.update_state({object_type + "_dirty" : False,
                           object_type + "_cleaned" : True})

    def encode_fillliquid(self, object_type : str, liquid : str):
        self.update_state({"empty_" + object_type : False,
                           object_type + "_filled_with_" + liquid : True})

    def encode_emptyliquid(self, object_type : str, liquid : str):
        self.update_state({"empty_" + object_type : True,
                           object_type + "_filled_with_" + liquid : False})

#r = RyeManager()
#
#pattern = "!(P({stoveburner_on : True} && Y({stoveburner_on : True}) && Y(Y({stoveburner_on : True})) && Y(Y(Y({stoveburner_on : True})))))"
#
#r.analysis(pattern)
#
#r.encode_pick("apple", "countertop")
#
#r.encode_put("apple", "plate")
#
#r.encode_drop("apple", "floor")
#
#r.encode_throw("apple", "floor")
#
#r.encode_moveheldback()
#
#r.encode_moveheldleft()
#
#r.encode_moveheldright()
#
#r.encode_moveheldup()
#
#r.encode_moveheldown()
#
#r.encode_pour("mug", "coffee")
#
#r.encode_push("fridge")
#
#r.encode_pull("fridge")
#
#r.encode_open("drawer")
#
#r.encode_close("microwave")
#
#r.encode_break("vase")
#
#r.encode_cook("bread")
#
#r.encode_slice("potato")
#
#r.encode_turnon("stoveburner")
#
#r.encode_turnoff("faucet")
#
#r.encode_dirty("plate")
#
#r.encode_clean("mug")
#
#r.encode_fillliquid("mug", "coffee")
#
#r.encode_emptyliquid("glass", "wine")
#
#r.save_to_json()