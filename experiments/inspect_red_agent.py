
import inspect

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)

env = CybORG(DroneSwarmScenarioGenerator(), "sim")

try:
    agent = env.environment_controller.agent_interfaces["red_agent_0"].agent

    print("Agent class:", type(agent).__name__)
    print("\nAction method:")
    print(inspect.getsource(type(agent).get_action))

    print("\nAgent initialization:")
    print(inspect.signature(type(agent).__init__))

finally:
    env.close()
