"""
清洗VLM输出文件,移除Markdown代码块等格式问题,直接覆盖原文件
"""

import json
from pathlib import Path
import re
import shutil


def clean_prediction(prediction: str) -> str:
    """清洗单个prediction字符串"""
    if not prediction:
        return prediction

    # 1. 移除Markdown代码块标记
    prediction = re.sub(r"```(?:python|json)?\s*\n?", "", prediction)

    # 2. 去除首尾空白
    prediction = prediction.strip()

    # 3. 尝试解析并重新序列化,统一格式(移除转义字符)
    try:
        # 先尝试直接解析
        parsed = json.loads(prediction)
        # 重新序列化为标准JSON格式,ensure_ascii=False避免Unicode转义
        prediction = json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        # 如果解析失败,尝试移除多余的反斜杠
        prediction = prediction.replace("\\n", "").replace('\\"', '"')
        try:
            parsed = json.loads(prediction)
            prediction = json.dumps(parsed, ensure_ascii=False)
        except:
            # 最后的备用清洗
            prediction = prediction.replace("\\", "")

    return prediction


def validate_prediction(prediction: str, num_segments: int) -> tuple[bool, str]:
    """验证prediction格式是否正确"""
    try:
        result = json.loads(prediction)

        if not isinstance(result, list):
            return False, f"不是列表格式: {type(result)}"

        if len(result) != num_segments:
            return False, f"列表长度错误: 期望{num_segments}, 实际{len(result)}"

        for i, item in enumerate(result):
            if not isinstance(item, str):
                return False, f"第{i}个元素不是字符串: {type(item)}"

        return True, "验证通过"

    except json.JSONDecodeError as e:
        return False, f"JSON解析失败: {e!s}"
    except Exception as e:
        return False, f"未知错误: {e!s}"


def clean_vlm_output_file(file_path: str, backup: bool = True):
    """清洗VLM输出文件并覆盖原文件

    Args:
        file_path: 文件路径
        backup: 是否创建备份文件 (默认True)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return

    print(f"处理文件: {file_path}")
    print("=" * 70)

    # 创建备份
    if backup:
        backup_path = file_path.parent / f"{file_path.stem}_backup.jsonl"
        shutil.copy2(file_path, backup_path)
        print(f"✓ 已创建备份: {backup_path}\n")

    # 读取并清洗数据
    cleaned_data = []
    error_count = 0
    cleaned_count = 0

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                episode_index = data["episode_index"]
                num_segments = data["num_segments"]
                original_prediction = data["prediction"]

                # 清洗prediction
                cleaned_prediction = clean_prediction(original_prediction)

                # 解析为真正的列表
                try:
                    prediction_list = json.loads(cleaned_prediction)
                except:
                    prediction_list = None
                    print(f"❌ Episode {episode_index} 无法解析prediction: {cleaned_prediction}")

                # 验证格式
                is_valid, error_msg = validate_prediction(cleaned_prediction, num_segments)

                # 显示清洗信息
                if cleaned_prediction != original_prediction:
                    cleaned_count += 1
                    print(f"Episode {episode_index} (已清洗):")
                    print(f"  原始: {original_prediction}")
                    print(f"  清洗: {cleaned_prediction}")

                    if not is_valid:
                        print(f"  ⚠ 警告: {error_msg}")
                        error_count += 1
                    print()

                elif not is_valid:
                    print(f"⚠ Episode {episode_index} 验证失败: {error_msg}")
                    print(f"  内容: {cleaned_prediction}")
                    print()
                    error_count += 1

                # 保存清洗后的数据
                cleaned_data.append(
                    {
                        "episode_index": episode_index,
                        "num_segments": num_segments,
                        "prediction": prediction_list,  # 保存为列表而不是字符串
                    }
                )

            except Exception as e:
                print(f"❌ 第{line_num}行处理失败: {e!s}")
                print(f"   内容: {line[:100]}...")
                error_count += 1

    # 覆盖原文件
    with open(file_path, "w", encoding="utf-8") as f:
        for data in cleaned_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # 统计信息
    print("=" * 70)
    print(f"总计处理: {len(cleaned_data)} 条")
    print(f"已清洗: {cleaned_count} 条")
    print(f"错误: {error_count} 条")
    print(f"✓ 已覆盖原文件: {file_path}")
    if backup:
        print(f"✓ 备份文件: {backup_path}")
    print("=" * 70)


if __name__ == "__main__":
    # ========== 配置 ==========
    from config import META_PATH

    file_path = f"{META_PATH}/aff_prompt_from_vlm.jsonl"
    # =========================

    clean_vlm_output_file(file_path, backup=True)
