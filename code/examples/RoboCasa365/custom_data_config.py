# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboCasa365 data config for GR00T N1.5 (QuantVLA v1.4, Stage D).

Key layout is dictated by the robocasa/robocasa365_checkpoints GR00T N1.5
checkpoints (target_posttraining): their saved new_embodiment statistics
(experiment_cfg/metadata.json) contain exactly five state groups and five
action groups, which this config mirrors:

  state:  base_position(3), base_rotation(4), end_effector_position_relative(3),
          end_effector_rotation_relative(4), gripper_qpos(2)        -> 16 dims
  action: gripper_close(1), end_effector_position(3), end_effector_rotation(3),
          base_motion(4), control_mode(1)                           -> 12 dims

padded by GR00TTransform to max_state_dim=64 / max_action_dim=32 (the
checkpoint's action_dim). Video = the three RoboCasa365 cameras under their
current namespace (video.robot0_agentview_left / _right / _eye_in_hand);
the wrapper also emits legacy video.res256_* aliases which this config
ignores.

Normalization stats come from the checkpoint's own metadata (loaded by
Gr00tPolicy), so no numbers are hardcoded here.
"""

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class RoboCasa365DataConfig(BaseDataConfig):
    video_keys = [
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    ]
    state_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
        "state.gripper_qpos",
    ]
    action_keys = [
        "action.gripper_close",
        "action.end_effector_position",
        "action.end_effector_rotation",
        "action.base_motion",
        "action.control_mode",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self) -> ModalityTransform:
        transforms = [
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "mean_std" for key in self.state_keys},
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "mean_std" for key in self.action_keys},
            ),
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
