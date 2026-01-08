import json
from pathlib import Path

from config import *
import pandas as pd


def read_gripper_events(gripper_events_path: str) -> list[dict]:
    """读取gripper事件jsonl文件

    Args:
        gripper_events_path: jsonl文件路径

    Returns:
        包含所有episode的gripper事件列表
    """
    episodes_data = []

    with open(gripper_events_path, encoding="utf-8") as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episodes_data.append(episode_data)

    return episodes_data


def get_episode_length(dataset_path: str, episode_index: int) -> int:
    """获取episode的总帧数

    Args:
        dataset_path: 数据集路径
        episode_index: episode索引

    Returns:
        该episode的总帧数
    """
    # 读取该episode的parquet文件
    episode_file = dataset_path / f"episode_{episode_index:06d}.parquet"

    if not episode_file.exists():
        raise FileNotFoundError(f"Episode文件不存在: {episode_file}")

    df = pd.read_parquet(episode_file)
    return len(df)


def generate_time_segments(gripper_changes: list[dict], total_frames: int, num_time_segments: int = 3) -> list[dict]:
    """根据夹爪变化生成时间段

    Args:
        gripper_changes: 夹爪变化事件列表
        total_frames: episode总帧数
        num_time_segments: 期望的时间段数量

    Returns:
        时间段列表，每个包含start_index, end_index, aff_prompt
    """
    # 提取所有gripper_step作为分割点
    split_points = [change["gripper_step"] for change in gripper_changes]

    # 根据num_time_segments决定使用多少个分割点
    max_splits = num_time_segments - 1

    if len(split_points) > max_splits:
        # 如果分割点太多，只取前max_splits个
        split_points = split_points[:max_splits]

    # 生成时间段
    segments = []
    start_idx = 0

    for i, split_point in enumerate(split_points):
        # 每个时间段：从start_idx到split_point
        segments.append(
            {
                "start_index": start_idx,
                "end_index": split_point,
                "aff_prompt": f"prompt{i+1}",  # 占位符
            }
        )
        start_idx = split_point + 1

    # 添加最后一个时间段：从最后一个分割点+1到最后一帧
    segments.append(
        {
            "start_index": start_idx,
            "end_index": total_frames - 1,  # 最后一帧索引
            "aff_prompt": f"prompt{len(segments)+1}",  # 占位符
        }
    )

    return segments


def generate_aff_prompts(
    dataset_name: str, task_name: str, num_time_segments: int = 3, output_path: str | None = None
) -> list[dict]:
    """生成所有episode的aff_prompts

    Args:
        dataset_name: 数据集名称
        task_name: 任务名称
        num_time_segments: 期望的时间段数量
        output_path: 输出jsonl文件路径，如果为None则不保存

    Returns:
        包含所有episode aff_prompts的列表
    """
    # 构建路径
    base_path = Path(f"/home/nightbreeze/research/Data/{dataset_name}")
    gripper_events_path = base_path / "gripper_analysis_results" / task_name / "gripper_events_all.jsonl"
    dataset_path = base_path / f"lerobotV2.1_data/{task_name}/data/chunk-000"
    output_path = base_path / f"lerobotV2.1_data/{task_name}/meta/first_aff_prompts_instruction.jsonl"

    episodes_gripper_data = read_gripper_events(str(gripper_events_path))

    # 为每个episode生成aff_prompts
    all_aff_prompts = []

    for episode_data in episodes_gripper_data:
        episode_index = episode_data["episode_index"]
        gripper_changes = episode_data["gripper_change"]

        print(f"\n处理 Episode {episode_index}:")
        print(f"  夹爪变化次数: {len(gripper_changes)}")

        # 获取该episode的总帧数
        try:
            total_frames = get_episode_length(dataset_path, episode_index)
            print(f"  总帧数: {total_frames}")
        except Exception as e:
            print(f"  错误: 无法获取episode长度 - {e}")
            continue

        # 生成时间段
        segments = generate_time_segments(gripper_changes, total_frames, num_time_segments)

        print(f"  生成 {len(segments)} 个时间段:")
        for i, seg in enumerate(segments, 1):
            print(f"    段{i}: 帧 {seg['start_index']} - {seg['end_index']} ({seg['aff_prompt']})")

        # 构建输出数据
        aff_prompt_data = {"episode_index": episode_index, "prompts_num": len(segments), "aff_prompts": segments}

        all_aff_prompts.append(aff_prompt_data)

    # 保存到jsonl文件
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n保存aff_prompts到: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for aff_prompt_data in all_aff_prompts:
                json_line = json.dumps(aff_prompt_data, ensure_ascii=False)
                f.write(json_line + "\n")

        print(f"成功保存 {len(all_aff_prompts)} 个episode的aff_prompts")

    return all_aff_prompts


def print_aff_prompts_summary(aff_prompts_data: list[dict]):
    """打印aff_prompts摘要信息"""
    print("\n" + "=" * 60)
    print("AFF Prompts 摘要:")
    print("=" * 60)

    for data in aff_prompts_data:
        episode_idx = data["episode_index"]
        prompts_num = data["prompts_num"]

        print(f"\nEpisode {episode_idx}: {prompts_num} 个时间段")
        for prompt_info in data["aff_prompts"]:
            start = prompt_info["start_index"]
            end = prompt_info["end_index"]
            prompt = prompt_info["aff_prompt"]
            frames = end - start + 1
            print(f"  {prompt}: 帧{start}-{end} ({frames}帧)")


def main():
    dataset_name = DATASET_NAME
    task_name = TASK_NAME
    num_segments = NUM_SEGMENTS

    print("=" * 60)
    print("生成 AFF Prompts")
    print("=" * 60)
    print(f"数据集: {dataset_name}")
    print(f"任务: {task_name}")
    print(f"期望时间段数量: {num_segments}")
    print("=" * 60)

    aff_prompts_data = generate_aff_prompts(
        dataset_name=dataset_name,
        task_name=task_name,
        num_time_segments=num_segments,
    )

    print_aff_prompts_summary(aff_prompts_data)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
