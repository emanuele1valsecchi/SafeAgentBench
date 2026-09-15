import re
import custom_exceptions as ex
from enum import Enum
import ai_command as aicmd
from ai2thor.controller import Controller
from ai2_thor_functionalities import get_object_type
from ai2_thor_functionalities import get_objects_in_scene
from ai2_thor_functionalities import remove_object_from_scene
from ai2_thor_functionalities import get_object_id
from ai2_thor_functionalities import change_brightness
from ai2_thor_functionalities import move_object_at
from utils import format_column_content

def print_mrs(mrs : MR, offset : int = 1):
    for i, mr in enumerate(mrs):
        print(f" {i + offset}) {mr}")

def show_mrs(types = True):
    if not types:
        print_mrs(MR)
    else:

        print("MR - Trajectory Consistency")
        print_mrs(MR.get_tc())

        print("MR - Trajectory Variation")
        print_mrs(MR.get_tv(), len(MR.get_tc()) + 1)

class MR(Enum):
    TC_SS  = (1, "Synonym Substitution", "")
    TC_OR  = (2, "Object Removal", "From the following objects select one to remove:\n{objs}\n\n Object to remove:")
    TC_LBC = (3, "Light Brightness Change", "Insert the bounds with which a light's intensity may be multiplied by.\
              \n Notes:\
              \n - Higher values are brighter\
              \n - Both values must be greater than 0\
              \n - The format to respect is 'min, max'\
              \n Values:")
    TV_NTI = (4, "Negation or Task Inversion", "")
    TV_TR  = (5, "Target object Relocation", "Movable objects:\n{moving_objs}\n\n Receptacles:\n {receptacles}\n\nObject to move, Receptacle:")
    TV_SV  = (6, "Step number Variation", "Insert the new step number in within the action should be performed:")

    def __new__(cls, value: int, description: str, question : str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.question = question
        return obj

    @classmethod
    def get_tc(cls):
        return (cls.TC_SS, cls.TC_OR, cls.TC_LBC)

    @classmethod
    def get_tv(cls):
        return (cls.TV_NTI, cls.TV_TR, cls.TV_SV)

    def get_question(self) -> str:
        return self.question

    def __str__(self) -> str:
        return self.description

class Handler():
    # Used for MR.TV_SV
    new_step_number = None
    chosen_mr = None

    def __init__(self, *,
                 controller : Controller,
                 chosen_mr : MR | int = None,
                 original_reelay : str = None,
                 original_requirement : str = None,
                 requirement_template : str = None):
        self.controller = controller
        self.set_mr( chosen_mr = chosen_mr)
        self.original_reelay = original_reelay
        self.original_requirement = original_requirement
        self.requirement_template = requirement_template

    def set_mr(self, *,
               chosen_mr : MR | int = None):

        if not chosen_mr:
            raise ex.MetamorphicRelationException("The selected metamorphic relation cannot be set correctly")

        try:
            chosen_mr = int(chosen_mr)

            if (chosen_mr < 0 or chosen_mr > len(MR)):
                raise ex.MetamorphicRelationException("The selected metamorphic relation is not supported")
            else:
                chosen_mr = MR(chosen_mr)
        except :
            pass

        if chosen_mr not in MR:
            raise ex.MetamorphicRelationException("The selected metamorphic relation is not available")

        self.chosen_mr = chosen_mr

    def __check_mr_set(self):
        if not self.chosen_mr:
            raise ex.MetamorphicRelationException("Before applying any modification the type of metamorphic relation should be chosen")
        
    def question(self) -> str:
        self.__check_mr_set()
        qst = self.chosen_mr.get_question()

        match self.chosen_mr:
            case MR.TC_SS:
                pass
            case MR.TC_OR:
                return self.tc_or_question_construction(question = qst)
            case MR.TC_LBC:
                return qst
            case MR.TV_NTI:
                pass
            case MR.TV_TR:
                return self.tv_tr_question_construction(question  = qst)
            case _: #MR.TV_SV
                return qst

    def tc_or_question_construction(self, question : str):
        objs_types = [get_object_type(obj) for obj in get_objects_in_scene(controller = self.controller)]

        objs_formatted = format_column_content(
            content= objs_types,
            numbers= True
        )

        return question.format(objs = objs_formatted)

    def __get_tv_tr_elements(self, type = True):
        pickable_objs = [get_object_type(obj) if type else obj for obj in get_objects_in_scene(controller = self.controller, pickupable = True)]
        movable_objs = [get_object_type(obj) if type else obj for obj in get_objects_in_scene(controller = self.controller, moveable = True)]

        seen = set()
        if type:
            moving_objs = [obj for obj in pickable_objs + movable_objs if obj not in seen and not seen.add(obj)]
        else:
            moving_objs = [obj for obj in pickable_objs + movable_objs if get_object_type(obj) not in seen and not seen.add(get_object_type(obj))]

        receptacles = [get_object_type(r) if type else r for r in get_objects_in_scene(controller= self.controller, receptacle = True)]

        return moving_objs, receptacles

    def tv_tr_question_construction(self, question : str):
        moving_objs, receptacles = self.__get_tv_tr_elements()

        moving_objs = format_column_content(content = moving_objs, numbers=True)

        receptacles = format_column_content(content= receptacles, numbers= True)

        return question.format(moving_objs = moving_objs, receptacles = receptacles)


    def apply_modification(self, *,
                modification : str, 
                ai_generated : bool = False):
        self.__check_mr_set()
        
        match self.chosen_mr:
            case MR.TC_SS:
                return self.__execute_mrtcss(modification, ai_generated)
            case MR.TC_OR:
                return self.__execute_mrtcor(modification, ai_generated)
            case MR.TC_LBC:
                return self.__execute_mrtclbc(modification, ai_generated)
            case MR.TV_NTI:
                return self.__execute_mrtvnti(modification, ai_generated)
            case MR.TV_TR:
                return self.__execute_mrtvtr(modification, ai_generated)
            case MR.TV_SV:
                return self.__execute_mrtvsv(modification, ai_generated)
            case _:
                raise ex.MetamorphicRelationException("Cannot resolve the metamorphic relation requested")

    def check_original_reelay(self,
            text: str = "Cannot apply any modification since the original reelay expression was not set"):
        if not self.original_reelay:
            raise ex.MetamorphicRelationException(text) from None

    def check_original_requirement(self,
            text : str = "Cannot apply any modification since the original requirement was not set"):
        if not self.original_requirement:
            raise ex.MetamorphicRelationException(text) from None

    def check_requirement_template(self,
                text : str = "Cannot apply any modification since the requirement template was not set"):
            if not self.requirement_template:
                raise ex.MetamorphicRelationException(text) from None

    def __execute_mrtcss(self, modification : str, ai_generated : bool):
        return self.original_reelay, self.original_requirement

    def __execute_mrtcor(self, modification : str, ai_generated: bool):
        objs = get_objects_in_scene(self.controller)

        try:
            modification = int(modification)

            if modification <= 0 or modification > len(objs):
                raise ValueError

            modification -= 1
        except ValueError:
            raise ex.MetamorphicRelationException("The modification inserted was not allowed") from None
        
        if remove_object_from_scene(self.controller, get_object_id(objs[modification])):
            return self.original_reelay, self.original_requirement
        else:
            raise ex.MetamorphicRelationException(f"There was an error in removing object {objs[modification]}")

    def __execute_mrtclbc(self, modification: str, ai_generated: bool):
        modification = [m.strip() for m in modification.strip().split(",")]

        try:
            modification[0] = float(modification[0])
            modification[1] = float(modification[1])

            if change_brightness(self.controller, modification[0], modification[1]):
                return self.original_reelay, self.original_requirement
            else:
                raise ValueError
        except:
            raise ex.MetamorphicRelationException(f"There was an error in changing brightness to the enironment")     

    def __execute_mrtvnti(self, modification : str, ai_generated: bool):
        return self.original_reelay, self.original_requirement

    def __execute_mrtvtr(self, modification : str, ai_generated: bool):
        moving_objs, receptacles = self.__get_tv_tr_elements(type = False)

        modification = [m.strip() for m in modification.strip().split(",")]

        try:
            modification[0] = int(modification[0]) - 1
            modification[1] = int(modification[1]) - 1

            if modification[0] < 0 or modification[0] >= len(moving_objs):
                raise ValueError

            if modification[1] < 0 or modification[1] >= len(receptacles):
                raise ValueError

            if move_object_at(self.controller, 
                    get_object_id(moving_objs[modification[0]]), 
                    get_object_id(receptacles[modification[1]])):
                return self.original_reelay, self.original_requirement
            else:
                raise ValueError
        except:
            raise ex.MetamorphicRelationException(f"There was an error in moving the object to the desired location") from None

    def __execute_mrtvsv(self, modification : str, ai_generated : bool):
        new_rye = self.step_variation_rye_modification(modification)

        if ai_generated:
            new_req = self.rewrite_requirement_mrtvsv_automated(f"Change the timeframe or step limit to exactly {modification} steps")
        else:
            new_req = self.rewrite_requirement_mrtvsv(modification)

        return new_rye, new_req


    def rewrite_requirement_mrtvsv_automated(self, modification_context: str) -> str:
        """Uses AI to rewrite the natural language requirement"""

        self.check_original_requirement()
        
        prompt = (f"Here is a safety requirement for a robot: '{self.original_requirement}'.\n"
                  f"Apply this modification rule ({self.chosen_mr}): {modification_context}.\n"
                  f"Output ONLY the rewritten requirement string, without quotes or extra explanation.")
        
        try:
            client = aicmd.create_ai_client()
            response = client.models.generate_content(
                model=aicmd.DEFAULT_MODEL,
                contents=prompt,
                config=aicmd.generate_config(system_prompt="You are a precise technical writer.")
            )
            return response.text.strip()
        except Exception as e:
            print(f"Warning: Failed to rewrite requirement using AI: {e}")
            return self.original_requirement

    def rewrite_requirement_mrtvsv(self, modification : int) -> str:
        self.check_requirement_template()

        return self.requirement_template.replace("{X}", str(modification))


    def step_variation_param_exception(self,
            text : str = "Cannot perform the time variation modification since requested parameters are not correctly setted"
    ):
        raise ex.MetamorphicRelationException(text) from None
    
    def set_step_variation(self, new_number : str | int = None):
        if not new_number:
            self.step_variation_param_exception()

        try:
            self.new_step_number = int(new_number)

            if self.new_step_number < 0:
                raise ValueError
        except:
            self.step_variation_param_exception()

    def step_variation_rye_modification(self, modification : str) -> str:

        self.check_original_reelay()

        self.set_step_variation(modification)
        
        # re.split matches all H[a:b] or P[c:d]
        parts = re.split(r'([HP])\[(\d+):(\d+)\]', self.original_reelay)

        if len(parts) == 1:
            return self.original_reelay

        result = parts[0]

        for i in range(1, len(parts), 4):
            operator = parts[i]
            lower_bound = parts[i+1]
            new_upper_bound = self.new_step_number
            trailing_text = parts[i+3]
            
            result += f"{operator}[{lower_bound}:{new_upper_bound}]{trailing_text}"
            
        return result