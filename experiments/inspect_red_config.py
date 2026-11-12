
import inspect
from pathlib import Path

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)

env = CybORG(DroneSwarmScenarioGenerator(), "sim")

try:
    agent = env.environment_controller.agent_interfaces[
        "red_agent_0"
    ].agent

    print("Agent class:", type(agent).__name__)
    print("Agent module:", type(agent).__module__)
    print("Agent source:", inspect.getfile(type(agent)))
    print("Constructor:", inspect.signature(type(agent).__init__))

    print("\nInitialization code:")
    print(inspect.getsource(type(agent).__init__))

    print("\nBehavior-related attributes:")
    for name, value in vars(agent).items():
        if any(word in name.lower() for word in ("chance", "behaviour", "target")):
            print(f"{name} = {value}")

finally:
    env.close()
