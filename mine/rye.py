import reelay
import json
import time

correct_sys_behavior = [
    dict(door_open=False, dow_suppressed=False, door_open_warning=False),
    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
    dict(door_open=True, dow_suppressed=False, door_open_warning=True),
    dict(door_open=True, dow_suppressed=True, door_open_warning=False),
    dict(door_open=True, dow_suppressed=True, door_open_warning=False),
]

#faulty_sys_behavior = [
#    dict(door_open=False,dow_suppressed=False, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=False, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=False, door_open_warning=True),
#    dict(door_open=True, dow_suppressed=True, door_open_warning=False),
#    dict(door_open=True, dow_suppressed=True, door_open_warning=True),
#    ]

faulty_sys_behavior = [
    dict(door_open=False,dow_suppressed=False, door_open_warning=False),
    dict(door_open=True),
    dict(),
    dict(),
    dict(),
    dict(),
    dict(door_open_warning=True),
    dict(dow_suppressed=True, door_open_warning=False),
    dict(door_open_warning=True),
]

#my_monitor_1 = reelay.discrete_timed_monitor(
#    pattern=r"(historically[0:5]{door_open} and not {dow_suppressed}) -> {door_open_warning}", condense=False)
#
#my_monitor_2 = reelay.discrete_timed_monitor(
#    pattern=r"{door_open_warning} -> historically[0:5]{door_open}", condense=False)
#
#my_monitor_3 = reelay.discrete_timed_monitor(
#    pattern=r"{door_open_warning} -> not {dow_suppressed}", condense=False)
#
#my_monitor_4 = reelay.discrete_timed_monitor(
#    pattern=r"{door_open_warning} -> not(pre({door_open} since {door_open_warning}))", condense=False)
#
#for x in faulty_sys_behavior:  # Change to correct_sys_behavior
#    r1 = my_monitor_1.update(x)
#    r2 = my_monitor_2.update(x)
#    r3 = my_monitor_3.update(x)
#    r4 = my_monitor_4.update(x)
#
#    if r1["value"] is False:
#        print('Error at {err_time} : False negative detected (SYS-REQ-01 Violation)'.format(err_time=my_monitor_1.now()))
#    if r2["value"] is False:
#        print('Error at {err_time} : False positive detected (SYS-REQ-01 Violation)'.format(err_time=my_monitor_2.now()))
#    if r3["value"] is False:
#        print('Error at {err_time} : False positive detected (SYS-REQ-01 Violation)'.format(err_time=my_monitor_3.now()))
#    if r4["value"] is False:
#        print('Error at {err_time} : False positive detected (SYS-REQ-02 Violation)'.format(err_time=my_monitor_4.now()))

class RyeManager:
    def __init__(self):
        self.store_data = []
        self.read_data = []

    def update_state(self, new_data : dict[str, str]):
        self.store_data.append(new_data)

    def save_to_json(self):
        with open("./dataset/sys_behavior.json", "w", encoding='utf-8') as f:
            json.dump(self.store_data, f, ensure_ascii=False, indent=4)

    def read_from_json(self):
        with open("./dataset/sys_behavior.json", "r") as f:
            self.read_data = json.load(f)

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

r = RyeManager()

r.encode_pick("apple", "countertop")

r.encode_put("apple", "plate")

r.encode_drop("apple", "floor")

r.encode_throw("apple", "floor")

r.encode_moveheldback()

r.encode_moveheldleft()

r.encode_moveheldright()

r.encode_moveheldup()

r.encode_moveheldown()

r.encode_pour("mug", "coffee")

r.encode_push("fridge")

r.encode_pull("fridge")

r.encode_open("drawer")

r.encode_close("microwave")

r.encode_break("vase")

r.encode_cook("bread")

r.encode_slice("potato")

r.encode_turnon("stoveburner")

r.encode_turnoff("faucet")

r.encode_dirty("plate")

r.encode_clean("mug")

r.encode_fillliquid("mug", "coffee")

r.encode_emptyliquid("glass", "wine")

r.save_to_json()

print(r.store_data)