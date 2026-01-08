import json
import cv2
import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from config import *

def extract_frames_from_videos(aff_prompts_path, video_dir_path, image_base_path):

    # 读取 JSONL 文件
    with open(aff_prompts_path, 'r') as f:
        lines = f.readlines()
    
    # 处理每个 episode
    for line in tqdm(lines, desc="Processing episodes"):
        data = json.loads(line.strip())
        episode_index = data['episode_index']
        prompts_num = data['prompts_num']
        aff_prompts = data['aff_prompts']
        
        # 构建视频路径 (假设视频命名为 episode_000000.mp4)
        video_filename = f"episode_{episode_index:06d}.mp4"
        video_path = os.path.join(video_dir_path, video_filename)
        
        # 检查视频是否存在
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # 构建图像保存目录
        episode_dir = f"episode_{episode_index:06d}"
        image_dir_path = os.path.join(image_base_path, episode_dir)
        
        # 创建目录(如果不存在)
        os.makedirs(image_dir_path, exist_ok=True)
        
        # 收集需要提取的帧索引
        frame_indices = [0]  # 第一帧 (start_index=0)
        for prompt in aff_prompts:
            frame_indices.append(prompt['end_index'])
        
        # 使用 ffmpeg 命令行工具提取帧（避免 AV1 硬件加速问题）
        saved_count = 0
        for frame_idx in frame_indices:
            image_filename = f"{saved_count}.png"
            image_path = os.path.join(image_dir_path, image_filename)
            
            # 使用 ffmpeg 提取特定帧，禁用硬件加速
            cmd = [
                'ffmpeg',
                '-hwaccel', 'none',  # 禁用硬件加速
                '-i', video_path,
                '-vf', f'select=eq(n\\,{frame_idx})',  # 选择特定帧
                '-vframes', '1',  # 只提取一帧
                '-y',  # 覆盖已存在的文件
                '-loglevel', 'error',  # 只显示错误信息
                image_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                saved_count += 1
            except subprocess.CalledProcessError as e:
                print(f"Error extracting frame {frame_idx} from {video_path}: {e.stderr}")
                continue
        
        # 验证是否保存了正确数量的图片
        expected_count = prompts_num + 1
        if saved_count != expected_count:
            print(f"Warning: Episode {episode_index} - Expected {expected_count} frames, saved {saved_count}")
        else:
            print(f"Episode {episode_index}: Saved {saved_count} frames to {image_dir_path}")
    
    print("Frame extraction completed!")

if __name__ == "__main__":

    first_aff_prompts_instruction_path = f"{META_PATH}/first_aff_prompts_instruction.jsonl"
    video_dir_path = VIDEO_PATH
    image_base_path = IMAGE_PATH

    extract_frames_from_videos(first_aff_prompts_instruction_path, video_dir_path, image_base_path)