
from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)

env = CybORG(DroneSwarmScenarioGenerator(), "sim")

try:
    agent_id = "red_agent_0"
    interface = env.environment_controller.agent_interfaces[agent_id]
    agent = interface.agent

    agent.chance_to_change = 0.8

    for step in range(1, 6):
        observation = env.get_observation(agent_id)
        action_space = env.get_action_space(agent_id)

        action = agent.get_action(observation, action_space)

        # Capture the state immediately before the action.
        before = env.get_true_state({})

        result = env.step(agent=agent_id, action=action)

        # Capture the state immediately after the action.
        after = env.get_true_state({})

        changed = str(before) != str(after)

        print(f"\nStep: {step}")
        print("Selected action:", type(action).__name__)
        print("Behavior:", getattr(agent, "behaviour_type", None))
        print("State representation changed:", changed)
        print("Reward:", getattr(result, "reward", None))

finally:
    env.close()
