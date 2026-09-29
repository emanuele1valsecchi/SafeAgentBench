import json
from utils import quit_program
from utils import print_separator
from utils import print_log
from agent_runner import get_scene
from agent_runner import get_instruction
from agent_runner import get_requirement
from agent_runner import get_reference_steps
from agent_runner import get_reelay_expression
from agent_runner import get_req_template
from agent_runner import get_default_step_number
from agent_runner import get_thesaurus_map
from agent_runner import get_inverted_reelay
from agent_runner import get_state_reelay_expression
from agent_runner import print_scenario
from agent_runner import define_task
from agent_runner import execute_rye_analysis
from agent_runner import execute_generated_plan_evaluation
from agent_runner import execute_mr_modification
from ai2_thor_functionalities import create_controller
from ai2_thor_functionalities import get_objects_in_scene
from ai_command import AiManager
from ai_command import aiEvaluator
from rye import RyeManager
from ai2_thor_executer import Ai2THORExecuter
from mr_handler import MR

REPEATS_PER_SCENARIO = 20
MR_TO_CHECK = len(MR) + 1

def get_scenarios():
    try:
        with open("./dataset/kitchen_tasks_and_constraints.json", "r") as f:
            pre_defined_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        quit_program(text = "Error in loading pre defined tasks. Please ensure the file exists in the specified path")

    return pre_defined_list

scenarios = get_scenarios()

completed_execution = 0
failed_execution = 0
success_judged = 0
fail_judged = 0
replanning_count = 0
total_inst_errors = 0
total_req_errors = 0
average_consistency = 0

for scenario_n in range(len(scenarios)):
    for _ in range(REPEATS_PER_SCENARIO):
        scenario = scenarios[scenario_n]
        scene = get_scene(scenario)
        instruction = get_instruction(scenario)
        requirement = get_requirement(scenario)
        reference_steps = get_reference_steps(scenario)
        reelay_expression = get_reelay_expression(scenario)

        req_template = get_req_template(scenario)
        default_X = get_default_step_number(scenario)
        thesaurus_map = get_thesaurus_map(scenario)
        inverted_reelay = get_inverted_reelay(scenario)
        state_reelay_expression = get_state_reelay_expression(scenario)

        print_scenario(
            title = f"Scenario {scenario_n + 1}",
            scene = scene,
            instruction = instruction,
            requirement = requirement,
            reference_steps = reference_steps,
            reelay_expression = reelay_expression
        )

        print_separator()
        print_log("Start Simulation")
        print_separator()

        controller = create_controller(scene=scene, width = 1280, height = 720)

        for mr_num in range(MR_TO_CHECK):
            environment_objects = get_objects_in_scene(controller)

            current_req = requirement
            current_reelay = reelay_expression

            if mr_num > 0:
                print_separator(title=f"Applying {MR(mr_num)}")
                current_reelay, current_req = execute_mr_modification(
                    controller=controller,
                    scene=scene,
                    instruction=instruction,
                    requirement=requirement,
                    reelay_expression=reelay_expression,
                    req_template=req_template,
                    default_X=default_X,
                    thesaurus_map=thesaurus_map,
                    inverted_reelay=inverted_reelay,
                    acquire_input=False,
                    chosen_mr=mr_num
                )
            else:
                print_separator(title="Baseline (no MR)")

            task, steps_ref = define_task(
                instruction = instruction,
                requirement = current_req,
                steps_ref = reference_steps,
                question = False
            )

            ai_manager = AiManager(
                reference_steps = steps_ref,
                task = task,
                environment_objects = environment_objects
            )

            rye_manager = RyeManager()

            ai_steps = ai_manager.resilient_generation_plan()

            if not ai_steps:
                print_log(f"Agent cannot generate an appropriate plan to execute {task}")
                continue

            executed = False

            executer = Ai2THORExecuter(
                controller = controller,
                plan = ai_steps,
                ai_manager = ai_manager,
                rye_manager = rye_manager
            )

            while not executed:
                print_log("Generated plan:")

                for i in range(len(executer.get_plan())):
                    print_log(f" {i + 1}) {executer.get_plan_step(i)}")

                print_separator()
                
                print_log("Executing plan: ")

                try:
                    executed = executer.execute_plan()
                except Exception as e:
                    print_log(text = str(e))
                    break

                if not executed:
                    print_separator()
                    print_log("\n Recreating the plan\n")
                    print_separator()

            print_separator()

            if executed:
                completed_execution += 1
                inst_errors, req_errors = execute_rye_analysis(
                    rye_manager = rye_manager,
                    req_reelay_expression = current_reelay,
                    inst_reelay_expression = state_reelay_expression,
                    acquire_input = False
                )

                (response, 
                 retries, 
                 planning_count, 
                 inst_errors_count, 
                 req_errors_count, 
                 consistency) = execute_generated_plan_evaluation(
                    controller = controller,
                    executer = executer,
                    ai_manager = ai_manager,
                    inst_errors = inst_errors,
                    req_errors = req_errors
                )

                if response == aiEvaluator.SUCCESS_CODE:
                    success_judged += 1
                elif response == aiEvaluator.FAILURE_CODE:
                    fail_judged += 1

                replanning_count += planning_count

                total_inst_errors += inst_errors_count
                total_req_errors += req_errors_count

                average_consistency += consistency
            else:
                failed_execution += 1

        print_separator( title = "Simulation complete")
        controller.stop()

if completed_execution > 0:
    average_consistency /= completed_execution
else:
    average_consistency = 0.0

print_separator( title = "Summary")
print_log(f"Total number of simulations: {failed_execution + completed_execution}")
print_log(f"Completed simulations: {completed_execution}")
print_log(f"Failed simulations: {failed_execution}")
print_log(f"Executions judged as successful: {success_judged}")
print_log(f"Executions judges as failure: {fail_judged}")
print_log(f"Total numbers of AI planning invocations: {failed_execution + completed_execution + replanning_count}")
print_log(f"Total errors in reelay expressions related to instructions: {total_inst_errors}")
print_log(f"Total errors in reelay expressions related to requirements: {total_req_errors}")
print_log(f"Average consistency: {average_consistency}")

