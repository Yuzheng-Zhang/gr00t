"""RealMan baseline: three RGB cameras and 58 joints, without tactile input.

action_semantics.json records source/data evidence. Preserve recorded units;
normalize each feature using training statistics. No unit conversion is performed.
"""
import json
from pathlib import Path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

JOINT_KEYS = ["left_arm", "left_hand", "right_arm", "right_hand"]
VIDEO_KEYS = ["left_wrist", "left_top", "right_wrist"]
ACTION_HORIZON = 40


def build_config(semantics=None):
    if semantics is None:
        semantics = json.loads(Path(__file__).with_name("action_semantics.json").read_text())
    action_configs = []
    for key in JOINT_KEYS:
        item = semantics[key]
        required = ["raw_action_kind", "state_unit", "action_unit", "representation"]
        missing = [name for name in required if not item.get(name)]
        if missing:
            raise ValueError(f"{key}: collection semantics not confirmed: {', '.join(missing)}")
        rep = ActionRepresentation[item["representation"]]
        expected = "step_delta" if rep == ActionRepresentation.DELTA else "absolute_position"
        if item["raw_action_kind"] != expected:
            raise ValueError(f"{key}: {rep.name} requires raw_action_kind={expected}")
        if rep == ActionRepresentation.RELATIVE and item["state_unit"] != item["action_unit"]:
            raise ValueError(f"{key}: relative conversion requires matching state/action units")
        action_configs.append(ActionConfig(
            rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT, state_key=key,
        ))
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=JOINT_KEYS),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)),
            modality_keys=JOINT_KEYS,
            action_configs=action_configs,
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"],
        ),
    }


def register_config():
    config = build_config()
    register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
    return config


# The official launcher imports the module, which registers the embodiment.
if __name__ != "__main__":
    realman_config = register_config()
else:
    build_config()
    print("RealMan modality configuration is complete.")
