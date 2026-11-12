import csv
import os
from collections import Counter

from CybORG import CybORG
from CybORG.Simulator.Scenarios.DroneSwarmScenarioGenerator import (
    DroneSwarmScenarioGenerator,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "training_data.csv",
)

ATTACKER_AGENT = "red_agent_0"
MONITOR_AGENT = "blue_agent_0"

EPISODES_PER_CONDITION = 5
STEPS_PER_EPISODE = 100

CONDITIONS = {
    "low_change": 0.0,
    "medium_change": 0.1,
    "high_change": 0.5,
}


# ============================================================
# ATTACK LABELS
# ============================================================

ATTACK_ACTIONS = {
    "ExploitDroneVulnerability",
    "SeizeControl",
    "FloodBandwidth",
    "BlockTraffic",
}

BENIGN_ACTIONS = {
    "Sleep",
}


def get_action_name(action):
    """Return CybORG action class name."""
    if action is None:
        return "None"

    return type(action).__name__


def action_to_label(action_name):
    """
    Ground-truth label.

    1 = attack
    0 = benign
    -1 = unknown
    """

    if action_name in ATTACK_ACTIONS:
        return 1

    if action_name in BENIGN_ACTIONS:
        return 0

    return -1


# ============================================================
# OBSERVATION FEATURES
# ============================================================

def extract_features(observation):
    """
    Extract features that a defender can observe.

    IMPORTANT:
    We deliberately do NOT use:
        selected_action
        behavior_type
        change_probability

    Those are attacker/internal information and would cause
    data leakage if used as ML features.
    """

    features = {
        "visible_hosts": 0,
        "interface_count": 0,
        "system_info_count": 0,
        "process_count": 0,
        "session_count": 0,
        "red_session_count": 0,
        "observation_keys": 0,
        "observation_size": 0,
        "success": -1,
    }

    if not isinstance(observation, dict):
        return features

    features["observation_keys"] = len(observation)

    # Numeric representation of the amount of observation data.
    features["observation_size"] = len(str(observation))

    # CybORG normally provides success as a special field.
    success = observation.get("success")

    if success is not None:

    # CybORG TrinaryEnum:
    # UNKNOWN = 2
    # TRUE    = 1
    # FALSE   = 0

        try:
            success_value = int(success.value)

            if success_value in (0, 1, 2):
                features["success"] = success_value

        except (AttributeError, TypeError, ValueError):

            if isinstance(success, bool):
                features["success"] = int(success)

            elif isinstance(success, (int, float)):
                features["success"] = int(success)

    for key, value in observation.items():

        if key == "success":
            continue

        if not isinstance(value, dict):
            continue

        # Host/interface information
        if "Interface" in value:

            features["visible_hosts"] += 1

            interfaces = value.get("Interface")

            if isinstance(interfaces, list):
                features["interface_count"] += len(interfaces)

            elif interfaces is not None:
                features["interface_count"] += 1

        # System information
        if "System info" in value:
            features["system_info_count"] += 1
        if "Processes" in value:
            processes = value.get("Processes")

            if isinstance(processes, list):
                features["process_count"] += len(processes)

        if "Sessions" in value:
            sessions = value.get("Sessions")

            if isinstance(sessions, list):
                features["session_count"] += len(sessions)

                for session in sessions:
                    if isinstance(session, dict):
                        if session.get("Agent") == ATTACKER_AGENT:
                            features["red_session_count"] += 1
    return features


def calculate_delta(before, after):
    """
    Difference between observations before and after
    the environment step.
    """

    return {
        "host_delta": (
            after["visible_hosts"]
            - before["visible_hosts"]
        ),

        "interface_delta": (
            after["interface_count"]
            - before["interface_count"]
        ),

        "system_info_delta": (
            after["system_info_count"]
            - before["system_info_count"]
        ),

        "observation_size_delta": (
            after["observation_size"]
            - before["observation_size"]
        ),

        "observation_keys_delta": (
            after["observation_keys"]
            - before["observation_keys"]
        ),
        "process_count": after["process_count"] - before["process_count"],
        "session_count": after["session_count"] - before["session_count"],
        "red_session_count": after["red_session_count"] - before["red_session_count"],
    }


# ============================================================
# GET RED AGENT
# ============================================================

def get_red_agent(env):
    """
    Get the actual RedDroneWormAgent object.

    In this CybORG version:

        env.agents
            -> list of agent names

        env.environment_controller.agent_interfaces
            -> contains the interfaces for red_agent_0

    The agent object is attached to the corresponding
    AgentInterface.
    """

    interface = env.environment_controller.agent_interfaces[
        ATTACKER_AGENT
    ]

    # Different CybORG versions expose the underlying agent
    # slightly differently, so handle the common forms.

    if hasattr(interface, "agent"):
        return interface.agent

    if hasattr(interface, "_agent"):
        return interface._agent

    raise RuntimeError(
        "Could not retrieve the Red agent object from "
        "the AgentInterface. Available attributes:\n"
        + str(dir(interface))
    )


# ============================================================
# GET RED ACTION AFTER STEP
# ============================================================

def get_executed_red_action(env):
    """
    Retrieve the action that CybORG actually executed
    for red_agent_0.
    """

    controller = env.environment_controller

    action_dict = getattr(
        controller,
        "action",
        None,
    )

    if isinstance(action_dict, dict):

        red_action = action_dict.get(
            ATTACKER_AGENT
        )

        if red_action is not None:
            return red_action

    return None


# ============================================================
# MAIN DATASET GENERATION
# ============================================================

def generate_dataset():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    rows = []

    print("=" * 60)
    print("ARGUS ML DATASET GENERATOR")
    print("=" * 60)

    print(f"Attacker : {ATTACKER_AGENT}")
    print(f"Monitor  : {MONITOR_AGENT}")
    print(
        f"Episodes : {EPISODES_PER_CONDITION}"
    )
    print(
        f"Steps    : {STEPS_PER_EPISODE}"
    )

    for condition_name, change_probability in CONDITIONS.items():

        print("\n" + "=" * 60)
        print(f"Condition: {condition_name}")
        print(
            f"Change probability: "
            f"{change_probability}"
        )
        print("=" * 60)

        for episode in range(
            EPISODES_PER_CONDITION
        ):

            print(
                f"\nEpisode "
                f"{episode + 1}/"
                f"{EPISODES_PER_CONDITION}"
            )

            # ------------------------------------------------
            # CREATE FRESH ENVIRONMENT
            # ------------------------------------------------

            scenario_generator = (
                DroneSwarmScenarioGenerator(
                    starting_num_red=0,

                    agent_to_drone_mapping={
                        0: "red"
                    },

                    maximum_steps=(
                        STEPS_PER_EPISODE
                    ),
                )
            )

            env = CybORG(
                scenario_generator,
                "sim",
            )

            # ------------------------------------------------
            # GET ACTUAL RED AGENT
            # ------------------------------------------------

            red_agent = get_red_agent(env)

            # Configure attacker behaviour change.
            red_agent.chance_to_change = (
                change_probability
            )

            print(
                f"Red agent: {red_agent.__class__.__name__}"
            )

            print(
                f"Initial behavior type: "
                f"{getattr(red_agent, 'behaviour_type', 'N/A')}"
            )

            print(
                f"Change probability: "
                f"{red_agent.chance_to_change}"
            )

            # ------------------------------------------------
            # INITIAL BLUE OBSERVATION
            # ------------------------------------------------

            previous_observation = (
                env.get_observation(
                    MONITOR_AGENT
                )
            )

            previous_action_name = "None"

            # ------------------------------------------------
            # RUN EPISODE
            # ------------------------------------------------

            for step in range(
                STEPS_PER_EPISODE
            ):

                # ============================================
                # OBSERVATION BEFORE ATTACK
                # ============================================

                before = extract_features(
                    previous_observation
                )

                # ============================================
                # LET RED GENERATE ITS ACTION
                # ============================================

                action_space = (
                    env.environment_controller
                    .agent_interfaces[
                        ATTACKER_AGENT
                    ]
                    .action_space
                )

                red_action = red_agent.get_action(
                    previous_observation,
                    action_space,
                )

                selected_action_name = (
                    get_action_name(
                        red_action
                    )
                )

                # ============================================
                # EXECUTE RED ACTION
                # ============================================

                result = env.step(
                    agent=ATTACKER_AGENT,
                    action=red_action,
                )

                # ============================================
                # GET BLUE OBSERVATION AFTER ATTACK
                # ============================================

                after_observation = result.observation

                after = extract_features(
                    after_observation
                )

                # ============================================
                # GET ACTUALLY EXECUTED RED ACTION
                # ============================================

                executed_red_action = (
                    get_executed_red_action(
                        env
                    )
                )

                executed_action_name = (
                    get_action_name(
                        executed_red_action
                    )
                )

                # If controller did not expose the action,
                # fall back to the action we explicitly gave it.
                if executed_red_action is None:
                    executed_action_name = (
                        selected_action_name
                    )

                # ============================================
                # GROUND TRUTH
                # ============================================

                label = action_to_label(
                    executed_action_name
                )

                # ============================================
                # FEATURE DELTAS
                # ============================================

                delta = calculate_delta(
                    before,
                    after
                )

                # ============================================
                # SAVE RECORD
                # ============================================

                row = {

                    # ----------------------------
                    # Time
                    # ----------------------------

                    "episode": episode,
                    "step": step,

                    # ----------------------------
                    # Observable features
                    # ----------------------------

                    "visible_hosts_before": (
                        before[
                            "visible_hosts"
                        ]
                    ),

                    "visible_hosts_after": (
                        after[
                            "visible_hosts"
                        ]
                    ),

                    "host_delta": (
                        delta[
                            "host_delta"
                        ]
                    ),

                    "interface_count_before": (
                        before[
                            "interface_count"
                        ]
                    ),

                    "interface_count_after": (
                        after[
                            "interface_count"
                        ]
                    ),

                    "interface_delta": (
                        delta[
                            "interface_delta"
                        ]
                    ),

                    "system_info_count_before": (
                        before[
                            "system_info_count"
                        ]
                    ),

                    "system_info_count_after": (
                        after[
                            "system_info_count"
                        ]
                    ),

                    "system_info_delta": (
                        delta[
                            "system_info_delta"
                        ]
                    ),

                    "observation_keys_before": (
                        before[
                            "observation_keys"
                        ]
                    ),

                    "observation_keys_after": (
                        after[
                            "observation_keys"
                        ]
                    ),

                    "observation_keys_delta": (
                        delta[
                            "observation_keys_delta"
                        ]
                    ),

                    "observation_size_before": (
                        before[
                            "observation_size"
                        ]
                    ),

                    "observation_size_after": (
                        after[
                            "observation_size"
                        ]
                    ),

                    "observation_size_delta": (
                        delta[
                            "observation_size_delta"
                        ]
                    ),

                    "success_before": (
                        before[
                            "success"
                        ]
                    ),

                    "success_after": (
                        after[
                            "success"
                        ]
                    ),

                    # ----------------------------
                    # Temporal feature
                    # ----------------------------

                    "previous_action": (
                        previous_action_name
                    ),

                    # ----------------------------
                    # Ground truth
                    # ----------------------------

                    "label": label,

                    # ----------------------------
                    # Metadata
                    #
                    # KEEP THESE FOR ANALYSIS,
                    # BUT DO NOT USE THEM AS
                    # ML INPUT FEATURES.
                    # ----------------------------

                    "selected_action": (
                        selected_action_name
                    ),

                    "executed_action": (
                        executed_action_name
                    ),

                    "behavior_type": (
                        getattr(
                            red_agent,
                            "behaviour_type",
                            None,
                        )
                    ),

                    "condition": (
                        condition_name
                    ),

                    "change_probability": (
                        change_probability
                    ),

                    # ----------------------------
                    # Reward
                    # ----------------------------

                    "reward": (
                        result.reward
                    ),
                    "process_count_before": before["process_count"],
                    "process_count_after": after["process_count"],
                    "process_delta": delta["process_count"],

                    "session_count_before": before["session_count"],
                    "session_count_after": after["session_count"],
                    "session_delta": delta["session_count"],

                    "red_session_count_before": before["red_session_count"],
                    "red_session_count_after": after["red_session_count"],
                    "red_session_delta": delta["red_session_count"],
                }

                rows.append(row)

                # Update state for next timestep.
                previous_observation = (
                    after_observation
                )

                previous_action_name = (
                    executed_action_name
                )

            print(
                f"  collected "
                f"{STEPS_PER_EPISODE} records"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = [
        "episode",
        "step",

        "visible_hosts_before",
        "visible_hosts_after",
        "host_delta",

        "interface_count_before",
        "interface_count_after",
        "interface_delta",

        "system_info_count_before",
        "system_info_count_after",
        "system_info_delta",

        "observation_keys_before",
        "observation_keys_after",
        "observation_keys_delta",

        "observation_size_before",
        "observation_size_after",
        "observation_size_delta",

        "success_before",
        "success_after",

        "previous_action",

        "label",

        # Metadata
        "selected_action",
        "executed_action",
        "behavior_type",
        "condition",
        "change_probability",

        "process_count_before",
        "process_count_after",
        "process_delta",

        "session_count_before",
        "session_count_after",
        "session_delta",

        "red_session_count_before",
        "red_session_count_after",
        "red_session_delta",

        "reward",
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    label_counts = Counter(
        row["label"]
        for row in rows
    )

    action_counts = Counter(
        row["executed_action"]
        for row in rows
    )

    success_counts = Counter(
        row["success_after"]
        for row in rows
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Total records: {len(rows)}"
    )

    print(
        f"Attack records: "
        f"{label_counts.get(1, 0)}"
    )

    print(
        f"Benign records: "
        f"{label_counts.get(0, 0)}"
    )

    print(
        f"Unknown records: "
        f"{label_counts.get(-1, 0)}"
    )

    print("\nExecuted actions:")

    for action, count in action_counts.items():
        print(
            f"  {action}: {count}"
        )

    print("\nSuccess values:")

    for value, count in success_counts.items():
        print(
            f"  {value}: {count}"
        )

    print("\nDataset saved to:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    generate_dataset()