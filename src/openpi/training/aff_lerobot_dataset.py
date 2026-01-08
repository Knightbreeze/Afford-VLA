"""
自定义 LeRobotDataset,支持 aff_prompts
"""

import json
import logging
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class AffLeRobotDataset(LeRobotDataset):
    """扩展 LeRobotDataset 支持 aff_prompts"""

    def __init__(
        self,
        repo_id: str,
        root=None,
        episodes=None,
        image_transforms=None,
        delta_timestamps=None,
        tolerance_s=1e-4,
        revision=None,
        force_cache_sync=False,
        download_videos=True,
        video_backend=None,
    ):
        # 调用父类初始化
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend,
        )

        # 加载 aff_prompts
        self.aff_prompts_data = None
        aff_prompts_path = "meta/aff_prompts.jsonl"
        aff_prompts_file = self.root / aff_prompts_path
        if aff_prompts_file.exists():
            self.aff_prompts_data = self._load_aff_prompts(aff_prompts_file)
            logging.info(f"Loaded {len(self.aff_prompts_data)} aff_prompts")

    def _load_aff_prompts(self, prompts_path):
        """加载 aff_prompts

        JSONL 格式示例：
        {"episode_index": 0, "prompts_num": 2, "aff_prompts": [
            {"start_index": 0, "end_index": 62, "aff_prompt": "prompt1"},
            {"start_index": 63, "end_index": 164, "aff_prompt": "prompt2"}
        ]}
        """
        prompts_path = Path(prompts_path)
        prompts_dict = {}

        # 读取 JSONL 文件
        with open(prompts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                episode_data = json.loads(line)
                episode_index = episode_data["episode_index"]

                # 为每个 aff_prompt 的帧范围分配 prompt
                for prompt_info in episode_data["aff_prompts"]:
                    start_idx = prompt_info["start_index"]
                    end_idx = prompt_info["end_index"]
                    aff_prompt = prompt_info["aff_prompt"]

                    # 为范围内的每一帧分配该 prompt
                    for frame_idx in range(start_idx, end_idx + 1):
                        prompts_dict[(episode_index, frame_idx)] = aff_prompt

        return prompts_dict

    def __getitem__(self, idx):
        # 调用父类的 __getitem__
        item = super().__getitem__(idx)

        # 添加 aff_prompt
        if self.aff_prompts_data is not None:
            ep_idx = item["episode_index"].item()
            frame_idx = item["frame_index"].item()

            prompt_key = (ep_idx, frame_idx)
            item["aff_prompt"] = self.aff_prompts_data[prompt_key]

        return item
