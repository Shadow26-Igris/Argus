
from CybORG.CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)

scenario = DroneSwarmScenarioGenerator()
env = CybORG(scenario, "sim")

agent_id = "blue_agent_0"
action_space = env.get_action_space(agent_id)

action_class = next(
    action
    for action, available in action_space["action"].items()
    if available and action.__name__ == "BlockTraffic"
)

ip_addresses = [
    ip for ip, available in action_space["ip_address"].items()
    if available
]

sessions = [
    session for session, available in action_space["session"].items()
    if available
]

print("Agent:", agent_id)
print("Available IP addresses:", len(ip_addresses))
print("Available sessions:", sessions)

if ip_addresses and sessions:
    action = action_class(
        session=sessions[0],
        agent=agent_id,
        ip_address=ip_addresses[0],
    )
    result = env.step(agent=agent_id, action=action)
    print("Action executed.")
    print("Reward:", getattr(result, "reward", None))
    print("Done:", getattr(result, "done", None))
else:
    print("No valid IP/session combination was exposed.")

env.close()
