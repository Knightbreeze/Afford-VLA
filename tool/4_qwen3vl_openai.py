import base64
import json
import os
import time

from config import *
from openai import OpenAI


def encode_image_to_base64(image_path):
    """将本地图片编码为 base64 字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_processer(image_paths):
    """构建图片内容(使用 base64 编码)"""
    image_contents = []
    for path in image_paths:
        if os.path.exists(path):
            base64_image = encode_image_to_base64(path)
            # 根据文件扩展名确定 MIME 类型
            ext = os.path.splitext(path)[1].lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"

            image_contents.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            )
        else:
            print(f"Warning: Image not found: {path}")
    return image_contents


def get_system_content(num_segments):
    # System Content: Establish perception logic and output rules
    system_content = """You are an embodied intelligence robot perception expert specializing in fine-grained manipulation tasks.
  Your goal is to identify the precise "Interaction Region" for each segment of the task based on the instruction AND the visual progression shown in the provided image sequence.

  ### Core Visual Analysis Guidelines

  1. Visual-Temporal Reasoning (CRITICAL):
    - Input Specification: You have {num_segments} + 1 images.
    - Mapping Rule:
      - Segment 1 Output -> Action between Image 0 and Image 1.
      - Segment 2 Output -> Action between Image 1 and Image 2.
    
    - CRITICAL DELTA ANALYSIS RULE: You MUST base your analysis SOLELY on the visual changes between Image i and Image i+1.
      - Contact Rule: If, in the transition from Image i to Image i+1, a gripper makes physical contact with an object (e.g., closes around it, touches its surface), the interaction region is the specific part being contacted.
      - Proximity & Intent Rule (FOR BLURRY/AMBIGUOUS IMAGES): If, in the transition from Image i to Image i+1, a gripper moves into extremely close proximity (within 1-2 cm visually) to a specific functional part of an object (e.g., handle, rim, button) AND this positioning is necessary for the next step of the task (as implied by the overall sequence), then treat this as an "interaction" with that part. Example: In Image 0, skillet is on table. In Image 1, gripper is positioned directly over the skillet handle, aligned for grasp, even if contact is not perfectly clear. Output: "the handle of the skillet".
      - If, in the transition from Image i to Image i+1, a gripper holding an object moves over and releases it into a container, the interaction region is the interior surface or opening of that container

  2. Dual-Arm / Multi-Object Logic:
    - Simultaneous Actions (VISUALLY CONFIRMED OR INTENDED): If, in the transition from Image i to Image i+1, two or more distinct grippers are observed to be in physical contact with two or more distinct objects at the same time, OR if they move into precise, functional positions relative to two distinct objects simultaneously (e.g., one gripper aligns with a handle while another aligns with a loaf), then this constitutes a simultaneous action.
      - Rule: Combine the descriptions using "and".
      - Example: If Left Hand grasps a "cup handle" and Right Hand grasps "bread" simultaneously -> Output: "the handle of the cup and the bread".
    - Complex Grasping: If the robot grasps a single large object with two hands -> Output: "the handles of the pot" or "both sides of the box".

  3. Granularity & Detail Rules:
    - Structured Objects (Tools, Containers): You MUST describe the functional part involved in the visual action (e.g., "handle", "rim", "button", "interior surface").
    - Uniform Objects: Describe as a whole (e.g., "apple", "sponge"), but retain visual attributes from the instruction/image (e.g., "the red apple", "the rough sponge").
    - Contextual Precision: Use adjectives (color, texture) to distinguish objects if multiple similar ones exist.

  4. Output Format:
    - Return strictly a Python list of strings. The list length must equal {num_segments}.
  """

    # User Content: Provide Chain-of-Thought (Few-Shot) Examples
    user_content_text = f"""

  ### Reference Examples (Visual-Temporal & Multi-Object Logic)

  Case 1 (Standard Tool Use - Detailed Description):
  Task: "Use the silver ladle to scoop soup from the black pot"
  Images: 3 (Img 0: Start, Img 1: Ladle held, Img 2: Ladle inside pot)
  Visual Analysis:
    - Seg 1 (Img 0->1): Robot hand contacts the ladle's handle. -> Focus: "the handle of the silver ladle"
    - Seg 2 (Img 1->2): Ladle tip moves into the liquid. -> Focus: "the liquid inside the black pot"
  Output: ["the handle of the silver ladle", "the liquid inside the black pot"]

  Case 2 (Dual-Arm / Simultaneous Grasping):
  Task: "Pick up the blue bottle and the red apple simultaneously"
  Images: 2 (Img 0: Objects on table, Img 1: Left hand holds bottle, Right hand holds apple)
  Visual Analysis:
    - Seg 1 (Img 0->1): Two interactions happen at once. Left arm targets bottle, Right arm targets apple.
  Output: ["the blue bottle and the red apple"]

  Case 3 (Complex Sequence with State Change):
  Task: "Move the gray skillet to the stove, then add the sausage into it"
  Images: 5 (Img 0: Start, Img 1: Skillet lifted, Img 2: Skillet on stove, Img 3: Sausage lifted, Img 4: Sausage in pan)
  Visual Analysis:
    - Seg 1 (Img 0->1): Gripper contacts pan handle. -> Focus: "the handle of the gray skillet"
    - Seg 2 (Img 1->2): Pan moves to stove burner. -> Focus: "the burner of the stove"
    - Seg 3 (Img 2->3): Gripper moves to sausage. -> Focus: "the sausage"
    - Seg 4 (Img 3->4): Sausage moves into the pan. -> Focus: "the interior surface of the gray skillet"
  Output: ["the handle of the gray skillet", "the burner of the stove", "the sausage", "the interior surface of the gray skillet"]

  Case 4 (Detailed Multi-Part):
  Task: "Put the golden croissant and the tea bag into the white mug"
  Images: 5 (Img 0->1: Croissant grasp, Img 1->2: Place in mug, Img 2->3: Tea bag grasp, Img 3->4: Place in mug)
  Output: ["the golden croissant", "the opening of the white mug", "the tea bag", "the opening of the white mug"]

  Case 5 (Simultaneous Grasp Before Transfer - WITH AMBIGUOUS CONTACT):
  Task: "Transfer the loaf into the skillet"
  Images: 3 (Img 0: Loaf and skillet on table, Img 1: Robot grips skillet handle with left hand AND loaf with right hand [Note: Image may be slightly blurry, but gripper is clearly aligned with handle], Img 2: Robot places loaf into skillet)
  Visual Analysis:
    - Seg 1 (Img 0->1): Two grippers move to position themselves for manipulation. Left gripper aligns precisely with the skillet handle, Right gripper contacts the loaf. Even if contact with handle is not perfectly sharp due to blur, the functional intent is clear. -> Interaction Region: "the handle of the skillet and the loaf".
    - Seg 2 (Img 1->2): Right gripper releases loaf into the skillet's interior. Left gripper maintains grip on handle for stability. -> Interaction Region: "the interior surface of the skillet".
  Output: ["the handle of the skillet and the loaf", "the interior surface of the skillet"]

  ### Current Task
  Instruction: "{instruction}"
  Number of Segments to generate: {num_segments}
  Input Images: {num_segments + 1} images provided.

  Please analyze the image transitions (Img i to Img i+1) to determine the exact interaction region.
  If two objects are manipulated at once, join them with "and".
  Output the List:
  """
    # 请你通过比较两个图片输出结果的变化，结合指令内容，输出每个步骤对应的交互区域描述，要求细粒度且符合规则。请给到我详细的分析流程

    return system_content, user_content_text


def get_image(image_dir, num_segments):
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        exit(1)

    import re

    image_files = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            match = re.match(r"(\d+)\.png", filename)
            if match:
                image_files.append((int(match.group(1)), filename))

    image_files.sort(key=lambda x: x[0])
    image_paths = [os.path.join(image_dir, filename) for _, filename in image_files]

    expected_count = num_segments + 1
    if len(image_paths) != expected_count:
        print(f"Error: Expected {expected_count} images, but found {len(image_paths)}")
        print(f"Required: num_segments ({num_segments}) + 1 = {expected_count}")
        exit(1)

    return image_paths


def get_instruction(tasks_path, episode_index):
    instruction = None
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["task_index"] == episode_index:
                instruction = data["task"]
                break
    if instruction is None:
        print(f"Error: Could not find task for episode_index {episode_index}")
        exit(1)
    return instruction


if __name__ == "__main__":
    # ----- 1. 初始化 OpenAI Client -----
    client = OpenAI(
        api_key=API_KEY,  # 请替换成您的ModelScope Access Token
        base_url=API_BASE_URL,
    )

    # ----- 2. 配置参数 -----
    tasks_path = f"{META_PATH}/tasks.jsonl"
    output_path = f"{META_PATH}/aff_prompt_from_vlm.jsonl"

    # ----- 3. 读取所有任务 -----
    all_tasks = []
    with open(tasks_path, encoding="utf-8") as f:
        for line in f:
            all_tasks.append(json.loads(line.strip()))

    print(f"Found {len(all_tasks)} episodes to process")
    print(f"Output will be saved to: {output_path}")
    print("=" * 70)

    # ----- 4. 批量处理所有 episode -----
    results = []

    for task_data in all_tasks:
        episode_index = task_data["task_index"]
        instruction = task_data["task"]

        print(f"\n[Episode {episode_index}/{len(all_tasks)-1}] Processing...")
        print(f"Instruction: {instruction}")

        try:
            image_dir = f"{IMAGE_PATH}/episode_{episode_index:06d}/"
            image_paths = get_image(image_dir, NUM_SEGMENTS)
            print(f"  ✓ Loaded {len(image_paths)} images")

            # 构建消息体
            image_contents = image_processer(image_paths)
            system_content, user_content_text = get_system_content(NUM_SEGMENTS)

            # 替换模板变量
            system_content = system_content.replace("{num_segments}", str(NUM_SEGMENTS))
            user_content_text = user_content_text.replace("{instruction}", instruction)
            user_content_text = user_content_text.replace("{num_segments}", str(NUM_SEGMENTS))

            # 调用模型
            response = client.chat.completions.create(
                model=MODEL_NAME,
                # model="Qwen/Qwen3-VL-32B-Instruct",
                messages=[
                    {"role": "system", "content": [{"type": "text", "text": system_content}]},
                    {"role": "user", "content": [*image_contents, {"type": "text", "text": user_content_text}]},
                ],
                stream=STREAM,
                temperature=TEMPERATURE,
            )

            model_output = response.choices[0].message.content
            print(f"  ✓ Model output: {model_output}")

            result = {"episode_index": episode_index, "num_segments": NUM_SEGMENTS, "prediction": model_output}
            results.append(result)

            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # 添加延迟避免API限流
            if episode_index < len(all_tasks) - 1:
                time.sleep(5)

        except Exception as e:
            print(f"  ✗ Error processing episode {episode_index}: {e!s}")
            # 记录错误但继续处理下一个
            result = {
                "episode_index": episode_index,
                "instruction": instruction,
                "num_segments": NUM_SEGMENTS,
                "prediction": None,
                "error": str(e),
            }
            results.append(result)
            continue

    # ----- 5. 输出统计信息 -----
    print("\n" + "=" * 70)
    print("Processing completed!")
    print(f"Total episodes: {len(all_tasks)}")
    print(f"Successful: {sum(1 for r in results if r.get('prediction') is not None)}")
    print(f"Failed: {sum(1 for r in results if r.get('prediction') is None)}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)
