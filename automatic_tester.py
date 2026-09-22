import subprocess
import time
from utils import print_separator

def run_simulation_test(test_name: str, simulated_inputs: list[str]):
    print_separator(title = f"Starting Test: {test_name}", save_to_log = False)
    
    input_string = "\n".join(simulated_inputs) + "\n"
    
    process = subprocess.run(
        ['python', 'agent_runnner.py'], 
        input=input_string,
        text=True,
        capture_output=True
    )
    
    with open(f"test_output_{test_name.replace(' ', '_')}.log", "w") as f:
        f.write("--- STDOUT ---\n")
        f.write(process.stdout)
        if process.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(process.stderr)

    print(f"Test '{test_name}' finished. Return code: {process.returncode}")
    print(f"Output saved to 'test_output_{test_name.replace(' ', '_')}.log'\n")

test_suites = [
    {
        "name": "Scenario 1 Baseline",
        "inputs": [
            "y",    # (Y/n) Do you want to load a pre defined scenario? :
            "1",    # Inser the scenario number to load
            "",     # Simulation complete. Press [Enter] to execute the rye analysis
            "n",    # (Y/n) Do you want to apply any Metamorphic Relation?
            ""      # Simulation complete. Press [Enter] to exit the program
        ]
    },
    {
        "name": "Scenario 1 with Metamorphic Relations",
        "inputs": [
            "y",        # (Y/n) Do you want to load a pre defined scenario? :
            "1",        # Inser the scenario number to load
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "y",        # (Y/n) Do you want to apply any Metamorphic Relation?
            "1",        # Chosen metamorphic relation
            "1, 1",     # Synonim Substitution. Values
            "",         # Press [Enter] to start the simulation with the updated scenario
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "y",        # (Y/n) Do you want to apply any Metamorphic Relation?
            "2",        # Chosen metamorphic relation
            "1",        # Object to remove
            "",         # Press [Enter] to start the simulation with the updated scenario
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "y",        # (Y/n) Do you want to apply any Metamorphic Relation?
            "3",        # Chosen metamorphic relation
            "0, 3",     # Light Intensity Values
            "",         # Press [Enter] to start the simulation with the updated scenario
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "y",        # (Y/n) Do you want to apply any Metamorphic Relation?
            "4",        # Chosen metamorphic relation
            "",         # Press [Enter] to execute the Negation or Task Inversion
            "",         # Press [Enter] to start the simulation with the updated scenario
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "y",        # (Y/n) Do you want to apply any Metamorphic Relation?
            "5",        # Chosen metamorphic relation
            "13, 39",   # Object to move, Receptacle
            "",         # Press [Enter] to start the simulation with the updated scenario
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "y",        # (Y/n) Do you want to apply any Metamorphic Relation?
            "6",        # Chosen metamorphic relation
            "5",        # Insert the new step number in within the action should be performed
            "",         # Press [Enter] to start the simulation with the updated scenario
            "",         # Simulation complete. Press [Enter] to execute the rye analysis
            "n",    # (Y/n) Do you want to apply any Metamorphic Relation?
            ""      # Simulation complete. Press [Enter] to exit the program
        ]
    },
]

for test in test_suites:
    run_simulation_test(test["name"], test["inputs"])
    time.sleep(2)