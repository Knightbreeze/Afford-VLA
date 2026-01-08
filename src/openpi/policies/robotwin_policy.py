import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_my_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation.state": np.random.rand(14),
        "rgb_right": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "rgb_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "rgb_global": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class RobotwinInputs(transforms.DataTransformFn):
    """Inputs for the Robotwin policy."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        data = _encode_state_my(data)

        inputs = {
            "state": data["observation.state"],
            "image": {
                "base_0_rgb": data["rgb_global"],
                "left_wrist_0_rgb": data["rgb_left"],
                "right_wrist_0_rgb": data["rgb_right"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = _encode_actions_my(actions)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "aff_prompt" in data:
            inputs["aff_prompt"] = data["aff_prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobotwinOutputs(transforms.DataTransformFn):
    """Outputs for the Robotwin policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": _decode_actions_my(actions)}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _encode_state_my(data: dict) -> dict:
    # state is [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # dim sizes: [6, 1, 6, 1]
    state = np.asarray(data["observation.state"])
    state[[6, 13]] = _normalize(state[[6, 13]], min_val=0, max_val=90)
    state[[6, 13]] = np.clip(state[[6, 13]], 0.0, 1.0)
    # state[[6, 13]] = _normalize(state[[6, 13]], min_val=0, max_val=90)
    # state[[6, 13]] = np.clip(state[[6, 13]], 0.0, 1.0)

    data["rgb_left"] = _parse_image(data["rgb_left"])
    data["rgb_right"] = _parse_image(data["rgb_right"])
    data["rgb_global"] = _parse_image(data["rgb_global"])

    data["obeservation.state"] = state

    return data


def _encode_actions_my(actions: np.ndarray) -> np.ndarray:
    actions[:, [6, 13]] = _normalize(actions[:, [6, 13]], min_val=0, max_val=90)
    actions[:, [6, 13]] = np.clip(actions[:, [6, 13]], 0.0, 1.0)
    actions = np.pad(actions, ((0, 0), (0, 32 - 14)), mode="constant", constant_values=0)
    return actions


def _decode_actions_my(actions: np.ndarray) -> np.ndarray:
    actions[:, [6, 13]] = _unnormalize(actions[:, [6, 13]], min_val=0, max_val=90)
    actions = actions[:, 0:14]
    return actions
