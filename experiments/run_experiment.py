
import csv
from pathlib import Path

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)

env = CybORG(DroneSwarmScenarioGenerator(), "sim")
agent_id = "blue_agent_0"
records = []

try:
    for step_number in range(1, 11):
        result = env.step(agent=agent_id)

        records.append({
            "step": step_number,
            "agent": agent_id,
            "action": str(getattr(result, "action", "unknown")),
            "reward": getattr(result, "reward", None),
            "done": getattr(result, "done", None),
        })

    output = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "review2_trajectory.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    rewards = [
        row["reward"] for row in records
        if isinstance(row["reward"], (int, float))
    ]

    print("Trajectory run completed.")
    print("Steps recorded:", len(records))
    print("Mean reward:", round(sum(rewards) / len(rewards), 2) if rewards else "N/A")
    print("Saved to:", output)

finally:
    env.close()
