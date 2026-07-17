Repository Structure — SafeAgentBench

Overview

This file summarizes the repository layout and the purpose of key files and folders.

Root
- **README.md**: Project overview, quickstart, and high-level descriptions.
- **requirements.txt**: Python dependencies (ai2thor, openai).
- **setup_project.sh**: Simple setup script (activates venv, exports an API key placeholder).

dataset/
- Contains JSONL benchmark splits:
  - **abstract_1009.jsonl**: Abstract tasks (4 instruction variants each), step sequences, risk categories, optional final_state.
  - **long_horizon_1009.jsonl**: Long-horizon multi-step safety scenarios.
  - **safe_detailed_1009.jsonl**: Detailed safe tasks.
  - **unsafe_detailed_1009.jsonl**: Detailed unsafe/hazardous tasks.

evaluator/
- LLM- and state-based evaluators for task success and safety:
  - **abstract_evaluate.py**: LLM-based judge for abstract plans (success/fail).
  - **detail_evaluate.py**: Compares environment object states to ground-truth final_state; LLM-based plan evaluation.
  - **long_horizon_evaluate.py**: LLM-based safety evaluation for long-horizon plans.
  - Google/GenAI variants: adapters for different LLM backends (e.g., detail_evaluate_google_genai.py, *_automated.py).

low_level_controller/
- Bridges high-level plan tokens to AI2-THOR actions:
  - **low_level_controller.py**: Single-agent `LowLevelPlanner` implementing high-level actions (find, pick, put, open/close, slice, turn on/off, drop, throw, break, cook, dirty, clean, fillLiquid, emptyLiquid, pour), navigation and retry logic.
  - **low_level_controller_multi_agent.py**: Multi-agent variant supporting per-agent state and `agentId`-aware calls.

methods/
- Example method implementations and helpers:
  - **map_vlm.py**: Vision+LLM planning example — converts images/table data into high-level plans and executes via low-level planner.
  - **utils.py**: Helpers for dataset loading, `gen_low_level_plan` (LLM conversion), `execute_low_level_plan`, LLM retry wrappers, and utilities.

figure/
- Assets used in the README and visualizations.

Notes
- Many components call external LLM APIs (OpenAI/GenAI); API keys are expected to be provided by the user/environment.
- `low_level_controller` is tightly coupled to AI2-THOR metadata and uses `env.step(...)` to run simulator actions.
- Treat the API keys in `setup_project.sh` as secrets; replace with secure management before use.

Next steps
- Run `pip install -r requirements.txt` and provide valid LLM and AI2-THOR setup to execute the scripts.
