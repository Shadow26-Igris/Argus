from pprint import pprint

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator
)

ATTACKER_AGENT = "red_agent_0"
MONITOR_AGENT = "blue_agent_0"


def main():

    scenario_generator = DroneSwarmScenarioGenerator(
        starting_num_red=0,
        agent_to_drone_mapping={0: "red"},
        maximum_steps=100
    )

    env = CybORG(scenario_generator, "sim")

    interface = env.environment_controller.agent_interfaces[ATTACKER_AGENT]
    red_agent = interface.agent

    print("\n========== INITIAL OBSERVATION ==========")

    before = env.get_observation(MONITOR_AGENT)

    print("Type:", type(before))
    print("Keys:", list(before.keys()))

    pprint(before)

    print("\n========== RED AGENT ==========")

    print("Agent:", red_agent)
    print("Behavior type:", red_agent.behaviour_type)
    print("Change probability:", red_agent.chance_to_change)

    print("\n========== RED ACTION ==========")

    red_action = red_agent.get_action(
        before,
        env.environment_controller.agent_interfaces[
            ATTACKER_AGENT
        ].action_space
    )

    print("Selected action:", red_action)

    print("\n========== EXECUTING STEP ==========")

    result = env.step(
        agent=ATTACKER_AGENT,
        action=red_action
    )

    print("Environment result:")
    print(result)

    print("\n========== AFTER OBSERVATION ==========")

    after = env.get_observation(MONITOR_AGENT)

    print("Type:", type(after))
    print("Keys:", list(after.keys()))

    pprint(after)

    print("\n========== OBSERVATION COMPARISON ==========")

    print("Before size:", len(str(before)))
    print("After size :", len(str(after)))

    print("Before keys:", list(before.keys()))
    print("After keys :", list(after.keys()))

    print("\nSuccess before:",
          before.get("success", "MISSING"))

    print("Success after:",
          after.get("success", "MISSING"))


if __name__ == "__main__":
    main()