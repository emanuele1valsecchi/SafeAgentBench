from ai2thor.controller import Controller
from ai_command import AiManager
from rye import RyeManager
from ai2_thor_task import ACTIONS
from ai2_thor_task import LIQUID
import ai2_thor_functionalities as func
import custom_exceptions as ex
from utils import print_log

class Ai2THORExecuter():
    def __init__(self, *, 
            controller : Controller = None,
            plan : list[str] = None,
            ai_manager : AiManager = None,
            rye_manager : RyeManager = None,
            replanning_count : int = 0
        ):
        self.controller = controller
        self.plan = plan
        self.ai_manager = ai_manager
        self.rye_manager = rye_manager
        self.new_plan = None
        self.replanning_count = replanning_count if replanning_count else 0

    # === UTILS === #
    def __check_controller(self):
        if not self.controller:
            raise ex.Ai2THORExecuterException("The Ai2-THOR controller was not set")

        return True
    
    def __check_plan(self):
        if not self.plan:
            raise ex.Ai2THORExecuterException("The Ai2-THOR plan was not set")

        return True

    def __check_ai_manager(self, raise_exception = False):
        if not self.ai_manager:
            if raise_exception:
                raise ex.Ai2THORExecuterException("The Ai2-THOR ai manager was not set")
            else:
                return False

        return True

    def __check_rye_manager(self, raise_exception = False):
        if not self.rye_manager:
            if raise_exception:
                raise ex.Ai2THORExecuterException("The Ai2-THOR reelay expression manager was not set")
            else:
                return False
        
        return True

    def __get_step_command(self, action : str, target : str, liquid : str):
        return f"{action} {func.get_object_type(target) if target else ''} {liquid or ''}".strip()

    def execute_plan(self):
        self.__check_controller()
        self.__check_plan()

        for i, step in enumerate(self.plan):
            action, target, liquid = self.__decode_step(step)
            
            if not action and not target and not liquid:
                raise ex.BadActionFormat(f"Error in command given to the agent")
    
            print_log(f"-> {self.__get_step_command(action, target, liquid)}")

            self.__perform_action(action, target, liquid)

            if self.__check_ai_manager():
                self.ai_manager.update_performed_actions(self.__get_step_command(action, target, liquid))

                if (i != (len(self.plan) - 1)):
                    self.new_plan = self.ai_manager.update_plan(self.plan, (i + 1), func.get_objects_in_scene(self.controller))
    
                    if (not (self.new_plan == self.plan)) and (not self.__is_plan_sublist(self.new_plan)):
                        self.replanning_count += 1
                        self.set_new_plan()
                        return False
    
        return True

    def __decode_step(self, step : str):
            # Extract action from the step to execute
            action = step.strip().split(" ", 1)[0].lower().strip()
    
            if not ACTIONS.is_action(action):
                raise ex.BadActionFormat(f"Action '{action}' was not recognized by the agent")
    
            action = ACTIONS(action)
    
            # Extract targets from the step to execute
            targets_id = [subject for subject in step.split()[1:]]
    
            if len(targets_id) != action.objects_required:
                raise ex.BadActionFormat(f"Action '{action}' requires exactly {action.objects_required} target(s), \
                    but {len(targets_id)} were provided")
    
            if action.objects_required == 0:
                return action, None, None

            target = self.__get_action_target(targets_id[0])
    
            if not target:
                raise ex.BadActionFormat(f"The target of '{action}' was not found in the scene")
            
    
            liquid = None
    
            if action.objects_required == 2:
                liquid = targets_id[1].lower()
                if not LIQUID.is_liquid(liquid):
                    raise ex.BadActionFormat(
                        f"The liquid '{liquid}' is not allowed, available liquids are: {LIQUID.get_all()}"
                    )
    
            return action, target, liquid

    def __get_action_target(self, target_id : str):
        target = func.get_object_by_id(self.controller, target_id)

        if target:
            return target
            
        target = func.get_object_by_name(self.controller, target_id)

        if target:
            return target

        target = func.get_object_by_type(self.controller, target_id)

        if target:
            return target

        return None

    def __perform_action(self, action : ACTIONS, target : dict[str, str], liquid : str):
        match action:
            case ACTIONS.FIND:
                func.reach_object(self.controller, target)

            case ACTIONS.PICK:
                func.pick_up_object(self.controller, target)

            case ACTIONS.PUT:
                func.put_object(self.controller, target)

            case ACTIONS.DROP:
                func.drop_object(self.controller)

            case ACTIONS.THROW:
                func.throw_object(self.controller)
                
            case ACTIONS.MOVEHELDBACK:
                func.move_held_object_back(self.controller)
                
            case ACTIONS.MOVEHELDLEFT:
                func.move_held_object_left(self.controller)
                
            case ACTIONS.MOVEHELDRIGHT:
                func.move_held_object_right(self.controller)
                
            case ACTIONS.MOVEHELDUP:
                func.move_held_object_up(self.controller)
                
            case ACTIONS.MOVEHELDDOWN:
                func.move_held_object_down(self.controller)
                
            case ACTIONS.POUR:
                func.rotate_held_object(self.controller)
                
            case ACTIONS.PUSH:
                func.directional_push_object(self.controller, target)
                
            case ACTIONS.PULL:
                func.direction_pull_object(self.controller, target)
                
            case ACTIONS.OPEN:
                func.open_object(self.controller, target)
                
            case ACTIONS.CLOSE:
                func.close_object(self.controller, target)
                
            case ACTIONS.BREAK:
                func.break_object(self.controller, target)
                
            case ACTIONS.COOK:
                func.cook_object(self.controller, target)
                
            case ACTIONS.SLICE:
                func.slice_object(self.controller, target)
                
            case ACTIONS.TURNON:
                func.toggle_object_on(self.controller, target)
                
            case ACTIONS.TURNOFF:
                func.toggle_object_off(self.controller, target)
                
            case ACTIONS.DIRTY:
                func.dirty_object(self.controller, target)
                
            case ACTIONS.CLEAN:
                func.clean_object(self.controller, target)
                
            case ACTIONS.FILLLIQUID:
                func.fill_object_with_liquid(self.controller, target, liquid)
                
            case ACTIONS.EMPTYLIQUID:
                func.empty_object_from_liquid(self.controller, target)
                
            case _:
                raise ex.BadActionFormat(f"Action '{action}' not allowed")

        if self.__check_rye_manager() : self.rye_manager.encode_scene_state(self.__get_step_command(action, target, liquid), self.controller.last_event.metadata)

    def __is_plan_sublist(self, new_plan : list):
        a_str = ','.join(map(str, self.plan))
        b_str = ','.join(map(str, new_plan))
    
        return a_str.find(b_str) != -1

    def set_new_plan(self, new_plan : list[str] = None):
        if not new_plan:
            new_plan = self.new_plan if self.new_plan else []
        
        self.plan = new_plan

    def get_plan(self):
        return self.plan

    def get_plan_step(self, i : int):
        return self.get_plan()[i]

    def get_replanning_count(self):
        return int(self.replanning_count)