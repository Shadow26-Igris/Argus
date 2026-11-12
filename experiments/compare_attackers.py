
import csv
from pathlib import Path

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "attacker_comparison.csv"
)


def run_experiment(label, change_probability, steps=100, episode=1):
    scenario_generator = DroneSwarmScenarioGenerator(
        starting_num_red=0,
        agent_to_drone_mapping={0: "red"}
    )

    env = CybORG(scenario_generator, "sim")
    records = []

    try:
        red_agent = env.environment_controller.agent_interfaces[
            "red_agent_0"
        ].agent
        red_agent.chance_to_change = change_probability

        print(f"\n=== {label} | change={change_probability} ===")

        for step in range(1, steps + 1):
            observation = env.get_observation("red_agent_0")
            action_space = env.get_action_space("red_agent_0")

            selected_action = red_agent.get_action(
                observation, action_space
            )

            result = env.step(
                agent="red_agent_0",
                action=selected_action
            )

            actual_action = getattr(result, "action", selected_action)
            returned_observation = getattr(
                result, "observation", {}
            )

            action_valid = (
                type(actual_action).__name__ != "InvalidAction"
            )

            print(
                f"step={step:02d} | "
                f"action={type(selected_action).__name__:<25} | "
                f"valid={action_valid} | "
                f"success={returned_observation.get('success')} | "
                f"reward={getattr(result, 'reward', None)}"
            )

            records.append({
                "condition": label,
                "change_probability": change_probability,
                "step": step,
                "selected_action": type(selected_action).__name__,
                "executed_action": type(actual_action).__name__,
                "action_valid": action_valid,

                "success_status": str(
                    returned_observation.get("success", "MISSING")
                ),
                "behavior_type": getattr(
                    red_agent, "behaviour_type", None
                ),
                "reward": getattr(result, "reward", None),
                "result_type": type(result).__name__,
                "done": getattr(result, "done", None),
                "episode": episode,
            })

            if getattr(result, "done", False):
                break

        print(f"Completed: {len(records)} steps")

        return records

    finally:
        env.close()


all_records = []
# all_records.extend(run_experiment("exploit_test", 0.0, steps=3))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

all_records = []

all_records.extend(
    run_experiment("low_change", 0.0, steps=100)
)

all_records.extend(
    run_experiment("medium_change", 0.1, steps=100)
)

all_records.extend(
    run_experiment("high_change", 0.5, steps=100)
)

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    fieldnames = [
    "episode",    
    "condition",
    "change_probability",
    "step",
    "selected_action",
    "executed_action",
    "action_valid",
    "success_status",
    "behavior_type",
    "reward",
    "result_type",
    "done",
    "target_ip",
    "session",
    "action_valid",
    "dropped",
    "blocked",
]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_records)

print("Comparison saved to:", OUTPUT)
