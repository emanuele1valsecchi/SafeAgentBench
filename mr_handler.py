import re
import custom_exceptions as ex
from enum import Enum
import ai_command as aicmd

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
    TC_OA  = (2, "Object Addition", "")
    TC_LBC = (3, "Light Brightness Change", "")
    TV_NTI = (4, "Negation or Task Inversion", "")
    TV_TR  = (5, "Target object Relocation", "")
    TV_SV  = (6, "Step number Variation", "Insert the new step number in within the action should be performed:")

    def __new__(cls, value: int, description: str, question : str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        obj.question = question
        return obj

    @classmethod
    def get_tc(cls):
        return (cls.TC_SS, cls.TC_OA, cls.TC_LBC)

    @classmethod
    def get_tv(cls):
        return (cls.TV_NTI, cls.TV_TR, cls.TV_SV)

    def get_question(self):
        return self.question

    def __str__(self) -> str:
        return self.description

class Handler():
    # Used for MR.TV_SV
    new_step_number = None

    def __init__(self, *,
                 chosen_mr : MR | int = None,
                 original_reelay : str = None,
                 original_requirement : str = None,
                 requirement_template : str = None):
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
        
    def question(self) -> tuple[str, ...]:
        self.__check_mr_set()
        return self.chosen_mr.get_question()


    def apply_modification(self, *,
                modification : str, 
                ai_generated : bool = False):
        self.__check_mr_set()
        
        match self.chosen_mr:
            case MR.TC_SS:
                return self.__execute_mrtcss(modification, ai_generated)
            case MR.TC_OA:
                return self.__execute_mrtcoa(modification, ai_generated)
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
        pass

    def __execute_mrtcoa(self, modification : str, ai_generated: bool):
        pass

    def __execute_mrtclbc(self, modification : str, ai_generated: bool):
        pass

    def __execute_mrtvnti(self, modification : str, ai_generated: bool):
        pass

    def __execute_mrtvtr(self, modification : str, ai_generated: bool):
        pass

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