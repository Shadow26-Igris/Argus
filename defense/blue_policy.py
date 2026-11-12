
def choose_blue_action(action_space, agent_id):
    """
    Select a defensive action exposed by CybORG.
    Prefer retaking control, then removing sessions,
    then blocking traffic. Fall back to Sleep.
    """
    preferred_actions = [
        "RetakeControl",
        "RemoveOtherSessions",
        "BlockTraffic",
        "AllowTraffic",
        "Sleep",
    ]

    available = action_space.get("action", {})

    for preferred in preferred_actions:
        for action_class, is_available in available.items():
            if is_available and action_class.__name__ == preferred:
                return action_class

    raise RuntimeError(f"No supported action found for {agent_id}")
