import ast
import json

from config import *


def process_prediction(prediction, spatial_rules=None):
    """
    处理prediction
    1. 如果是字符串则解析为列表
    2. 应用自定义空间规则

    Args:
        prediction: prediction数据（列表或字符串）
        spatial_rules: 规则字典，格式为：
            {
                "check_indices": [0, 1, 2],  # 要检查的位置索引
                "check_strings": ["handle", "surface", "rim"],  # 检查是否包含这些字符串
                "modify_strings": [" and the handle", "the interior surface", " and the rim"]  # 修改规则
            }
            - 如果 modify_strings[i] 以 "and " 或 " and " 开头，则追加到原值后面
            - 否则，直接覆盖为新值
    """
    # 如果已经是列表，直接使用；否则尝试解析
    if isinstance(prediction, list):
        prediction_list = prediction.copy()  # 创建副本避免修改原数据
    else:
        try:
            prediction_list = json.loads(prediction)
        except:
            try:
                prediction_list = ast.literal_eval(prediction)
            except:
                print(f"Warning: Failed to parse prediction: {prediction}")
                return None

    # 应用自定义规则
    if spatial_rules:
        check_indices = spatial_rules.get("check_indices", [])
        check_strings = spatial_rules.get("check_strings", [])
        modify_strings = spatial_rules.get("modify_strings", [])

        # 确保三个列表长度一致
        if len(check_indices) != len(check_strings) or len(check_indices) != len(modify_strings):
            print("  ⚠ Warning: Rule lists have different lengths, skipping rules")
            return prediction_list

        for i, idx in enumerate(check_indices):
            # 检查索引是否有效
            if idx >= len(prediction_list):
                print(f"  ⚠ Warning: Index {idx} out of range, skipping")
                continue

            current_value = prediction_list[idx]
            check_str = check_strings[i]
            modify_str = modify_strings[i]

            # 检查是否包含目标字符串（不区分大小写）
            if check_str.lower() not in current_value.lower():
                # 判断是追加还是覆盖
                if modify_str.startswith("and ") or modify_str.startswith(" and "):
                    # 追加模式
                    prediction_list[idx] = current_value + modify_str
                    print(f"  → Appended to prediction[{idx}]: {prediction_list[idx]}")
                else:
                    # 覆盖模式
                    prediction_list[idx] = modify_str
                    print(f"  → Replaced prediction[{idx}]: {prediction_list[idx]}")

    return prediction_list


def merge_aff_prompts(dataset_name, task_name, spatial_rules=None):
    """
    合并first_aff_prompts_instruction和aff_prompt_from_vlm
    生成最终的aff_prompts.jsonl

    Args:
        dataset_name: 数据集名称
        task_name: 任务名称
        spatial_rules: 空间规则字典（可选）
    """
    # 构建路径
    base_path = f"/home/nightbreeze/research/Data/{dataset_name}/lerobotV2.1_data/{task_name}/meta"
    first_aff_path = f"{base_path}/first_aff_prompts_instruction.jsonl"
    vlm_path = f"{base_path}/aff_prompt_from_vlm.jsonl"
    output_path = f"{base_path}/aff_prompts.jsonl"

    print("Reading from:")
    print(f"  - {first_aff_path}")
    print(f"  - {vlm_path}")
    print(f"Output to: {output_path}")
    print("=" * 70)

    # 读取first_aff_prompts_instruction.jsonl
    first_aff_data = {}
    with open(first_aff_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            first_aff_data[data["episode_index"]] = data

    # 读取aff_prompt_from_vlm.jsonl
    vlm_data = {}
    with open(vlm_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            vlm_data[data["episode_index"]] = data

    # 合并数据
    results = []

    for episode_index in sorted(first_aff_data.keys()):
        print(f"\n[Episode {episode_index}]")

        if episode_index not in vlm_data:
            print("  ✗ Warning: No VLM prediction found, skipping...")
            continue

        first_data = first_aff_data[episode_index]
        vlm_prediction = vlm_data[episode_index]["prediction"]

        # 处理prediction
        prediction_list = process_prediction(vlm_prediction, spatial_rules=spatial_rules)

        if prediction_list is None:
            print("  ✗ Failed to process prediction, skipping...")
            continue

        # 验证长度匹配
        if len(prediction_list) != first_data["prompts_num"]:
            print(
                f"  ✗ Warning: Prediction length ({len(prediction_list)}) != prompts_num ({first_data['prompts_num']})"
            )
            continue

        # 填充aff_prompt
        updated_aff_prompts = []
        for i, aff_prompt_item in enumerate(first_data["aff_prompts"]):
            updated_item = {
                "start_index": aff_prompt_item["start_index"],
                "end_index": aff_prompt_item["end_index"],
                "aff_prompt": prediction_list[i],
            }
            updated_aff_prompts.append(updated_item)
            print(f"  [{i}] {aff_prompt_item['start_index']}-{aff_prompt_item['end_index']}: {prediction_list[i]}")

        # 构建最终结果
        result = {
            "episode_index": episode_index,
            "prompts_num": first_data["prompts_num"],
            "aff_prompts": updated_aff_prompts,
        }
        results.append(result)

    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("Processing completed!")
    print(f"Total episodes processed: {len(results)}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    # ========== 配置空间规则（可选） ==========
    # 示例规则配置:
    # spatial_rules = {
    #     "check_indices": [0, 1],  # 检查第0和第1个元素
    #     "check_strings": ["handle", "surface"],  # 检查是否包含这些字符串
    #     "modify_strings": [" and the handle of the skillet", "the interior surface"]  # 修改规则
    # }

    # 针对 place_bread_skillet 任务的规则
    spatial_rules = SPATIAL_RULES

    # 如果不需要规则，设置为 None
    # spatial_rules = None
    # ==========================================

    # 执行合并
    merge_aff_prompts(DATASET_NAME, TASK_NAME, spatial_rules)
