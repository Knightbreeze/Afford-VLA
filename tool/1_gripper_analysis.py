import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from config import *

@dataclass
class GripperConfig:
    # 夹爪在observation.state中的索引
    left_gripper_index: int = LEFT_GRIPPER_INDEX
    right_gripper_index: int = RIGHT_GRIPPER_INDEX
    
    # 夹爪值的含义 (True: 1=打开,0=闭合; False: 1=闭合,0=打开)
    one_means_open: bool = True
    
    # 值的范围
    value_range: Tuple[float, float] = (0.0, 1.0)
    
    # 阈值设置
    open_threshold: float = OPEN_THRESHOLD   # 判断为打开的阈值
    close_threshold: float = CLOSE_THRESHOLD  # 判断为闭合的阈值
    
    # 检测参数
    min_duration: int = MIN_DURATION  # 最小持续帧数
    
    def __post_init__(self):
        if self.one_means_open:
            assert self.open_threshold > self.close_threshold, \
                "当1表示打开时,open_threshold应该大于close_threshold"
        else:
            assert self.close_threshold > self.open_threshold, \
                "当1表示闭合时,close_threshold应该大于open_threshold"
    
    def get_state_name(self, is_open: bool) -> str:
        return "open" if is_open else "closed"
    
    def print_config(self):
        print(f"\n{'='*60}")
        print("夹爪配置:")
        print(f"{'='*60}")
        print(f"左手夹爪索引: {self.left_gripper_index}")
        print(f"右手夹爪索引: {self.right_gripper_index}")
        print(f"值含义: {'1=打开, 0=闭合' if self.one_means_open else '1=闭合, 0=打开'}")
        print(f"值范围: [{self.value_range[0]}, {self.value_range[1]}]")
        print(f"打开阈值: {self.open_threshold} ({'≥' if self.one_means_open else '≤'} 此值认为打开)")
        print(f"闭合阈值: {self.close_threshold} ({'≤' if self.one_means_open else '≥'} 此值认为闭合)")
        print(f"最小持续帧数: {self.min_duration}")


def load_single_episode(file_path: str) -> pd.DataFrame:
    # print(f"读取文件: {file_path}")
    df = pd.read_parquet(file_path)
    return df


def extract_gripper_states(df: pd.DataFrame, config: GripperConfig) -> Tuple[np.ndarray, np.ndarray]:
    """从observation.state中提取左右手夹爪状态"""
    observation_state = np.stack(df['observation.state'].values)
    
    left_gripper = observation_state[:, config.left_gripper_index]
    right_gripper = observation_state[:, config.right_gripper_index]
    
    return left_gripper, right_gripper


def detect_gripper_changes(gripper_values: np.ndarray, 
                          config: GripperConfig) -> Dict:
    """检测夹爪状态变化
    
    Args:
        gripper_values: 夹爪位置值数组
        config: 夹爪配置
    
    Returns:
        包含状态变化信息的字典
    """
    # 根据配置判断夹爪状态
    if config.one_means_open:
        # 1=打开, 0=闭合
        is_open = gripper_values >= config.open_threshold
        is_closed = gripper_values <= config.close_threshold
    else:
        # 1=闭合, 0=打开
        is_closed = gripper_values >= config.close_threshold
        is_open = gripper_values <= config.open_threshold
    
    # 检测状态变化点
    open_state_changes = np.diff(is_open.astype(int))
    close_state_changes = np.diff(is_closed.astype(int))
    
    # 从闭到开
    open_points = np.where(open_state_changes == 1)[0] + 1
    # 从开到闭
    close_points = np.where(close_state_changes == 1)[0] + 1
    
    # 过滤掉持续时间太短的状态变化
    filtered_open = []
    filtered_close = []
    
    # 过滤打开事件
    for op in open_points:
        # 找到下一个闭合点
        next_close = close_points[close_points > op]
        duration = next_close[0] - op if len(next_close) > 0 else len(is_open) - op
        if duration >= config.min_duration:
            filtered_open.append(int(op))
    
    # 过滤闭合事件
    for cp in close_points:
        # 找到下一个打开点
        next_open = open_points[open_points > cp]
        duration = next_open[0] - cp if len(next_open) > 0 else len(is_closed) - cp
        if duration >= config.min_duration:
            filtered_close.append(int(cp))
    
    return {
        'values': gripper_values,
        'is_open': is_open,
        'is_closed': is_closed,
        'open_points': np.array(filtered_open, dtype=np.int32),
        'close_points': np.array(filtered_close, dtype=np.int32),
        'initial_state': 'open' if is_open[0] else 'closed',
        'final_state': 'open' if is_open[-1] else 'closed'
    }


def analyze_gripper_statistics(gripper_info: Dict, name: str, config: GripperConfig) -> None:
    """打印夹爪统计信息"""
    print(f"\n{'='*60}")
    print(f"{name} 统计信息:")
    print(f"{'='*60}")
    
    print(f"初始状态: {gripper_info['initial_state']}  最终状态: {gripper_info['final_state']}")
    print(f"打开次数: {len(gripper_info['open_points'])}  闭合次数: {len(gripper_info['close_points'])}")

    if len(gripper_info['open_points']) > 0:
        print(f"\n打开时间点 (帧索引):")
        for i, op in enumerate(gripper_info['open_points'], 1):
            op = int(op)
            print(f"  {i}. 帧 {op} (值: {gripper_info['values'][op]:.4f})")
    else:
        print("\n未检测到打开动作")
    
    if len(gripper_info['close_points']) > 0:
        print(f"\n闭合时间点 (帧索引):")
        for i, cp in enumerate(gripper_info['close_points'], 1):
            cp = int(cp)
            print(f"  {i}. 帧 {cp} (值: {gripper_info['values'][cp]:.4f})")
    else:
        print("\n未检测到闭合动作")
    
    # 计算持续时间统计
    if len(gripper_info['open_points']) > 0 and len(gripper_info['close_points']) > 0:
        open_durations = []
        for op in gripper_info['open_points']:
            next_close = gripper_info['close_points'][gripper_info['close_points'] > op]
            if len(next_close) > 0:
                open_durations.append(next_close[0] - op)
        
        if open_durations:
            print(f"\n打开持续时间统计:")
            print(f"  平均: {np.mean(open_durations):.2f} 帧")
            print(f"  最短: {np.min(open_durations)} 帧")
            print(f"  最长: {np.max(open_durations)} 帧")


def plot_gripper_states(df: pd.DataFrame, 
                       left_info: Dict, 
                       right_info: Dict,
                       config: GripperConfig,
                       save_path: str = "episode_gripper_analysis.png") -> None:
    """可视化夹爪状态"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    frame_indices = np.arange(len(df))
    
    # 确保索引是整数类型
    left_open_points = left_info['open_points'].astype(np.int32)
    left_close_points = left_info['close_points'].astype(np.int32)
    right_open_points = right_info['open_points'].astype(np.int32)
    right_close_points = right_info['close_points'].astype(np.int32)
    
    # 设置标签
    value_meaning = "1=Open, 0=Closed" if config.one_means_open else "1=Closed, 0=Open"
    
    # 左手夹爪值
    axes[0].plot(frame_indices, left_info['values'], 'b-', label='Left Gripper', linewidth=1.5)
    axes[0].axhline(y=config.open_threshold, color='g', linestyle='--', alpha=0.5, 
                   label=f'Open Threshold ({config.open_threshold})')
    axes[0].axhline(y=config.close_threshold, color='r', linestyle='--', alpha=0.5, 
                   label=f'Close Threshold ({config.close_threshold})')
    if len(left_open_points) > 0:
        axes[0].scatter(left_open_points, 
                       left_info['values'][left_open_points], 
                       color='green', s=150, marker='^', label='Open Event', zorder=5, edgecolors='black')
    if len(left_close_points) > 0:
        axes[0].scatter(left_close_points, 
                       left_info['values'][left_close_points], 
                       color='red', s=150, marker='v', label='Close Event', zorder=5, edgecolors='black')
    axes[0].set_ylabel(f'Gripper Value ({value_meaning})', fontsize=12)
    axes[0].set_title('Left Gripper State Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([config.value_range[0] - 0.1, config.value_range[1] + 0.1])
    
    # 右手夹爪值
    axes[1].plot(frame_indices, right_info['values'], 'r-', label='Right Gripper', linewidth=1.5)
    axes[1].axhline(y=config.open_threshold, color='g', linestyle='--', alpha=0.5, 
                   label=f'Open Threshold ({config.open_threshold})')
    axes[1].axhline(y=config.close_threshold, color='r', linestyle='--', alpha=0.5, 
                   label=f'Close Threshold ({config.close_threshold})')
    if len(right_open_points) > 0:
        axes[1].scatter(right_open_points, 
                       right_info['values'][right_open_points], 
                       color='green', s=150, marker='^', label='Open Event', zorder=5, edgecolors='black')
    if len(right_close_points) > 0:
        axes[1].scatter(right_close_points, 
                       right_info['values'][right_close_points], 
                       color='red', s=150, marker='v', label='Close Event', zorder=5, edgecolors='black')
    axes[1].set_ylabel(f'Gripper Value ({value_meaning})', fontsize=12)
    axes[1].set_title('Right Gripper State Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([config.value_range[0] - 0.1, config.value_range[1] + 0.1])
    
    # 组合视图
    axes[2].plot(frame_indices, left_info['values'], 'b-', label='Left Gripper', linewidth=2, alpha=0.7)
    axes[2].plot(frame_indices, right_info['values'], 'r-', label='Right Gripper', linewidth=2, alpha=0.7)
    axes[2].axhline(y=config.open_threshold, color='gray', linestyle='--', alpha=0.3)
    axes[2].axhline(y=config.close_threshold, color='gray', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Frame Index', fontsize=12)
    axes[2].set_ylabel(f'Gripper Value ({value_meaning})', fontsize=12)
    axes[2].set_title('Combined Gripper States', fontsize=14, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([config.value_range[0] - 0.1, config.value_range[1] + 0.1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results_to_csv(df: pd.DataFrame,
                       left_info: Dict,
                       right_info: Dict,
                       save_path: str = "episode_gripper_events.csv") -> None:
    """将夹爪事件保存到CSV文件"""
    events = []
    
    # 左手事件
    for op in left_info['open_points']:
        op = int(op)
        events.append({
            'frame_index': op,
            'gripper': 'left',
            'event': 'open',
            'value': float(left_info['values'][op])
        })
    
    for cp in left_info['close_points']:
        cp = int(cp)
        events.append({
            'frame_index': cp,
            'gripper': 'left',
            'event': 'close',
            'value': float(left_info['values'][cp])
        })
    
    # 右手事件
    for op in right_info['open_points']:
        op = int(op)
        events.append({
            'frame_index': op,
            'gripper': 'right',
            'event': 'open',
            'value': float(right_info['values'][op])
        })
    
    for cp in right_info['close_points']:
        cp = int(cp)
        events.append({
            'frame_index': cp,
            'gripper': 'right',
            'event': 'close',
            'value': float(right_info['values'][cp])
        })
    
    # 转换为DataFrame并排序
    if events:
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values('frame_index')
        events_df.to_csv(save_path, index=False)
        print(f"事件记录已保存到: {save_path}")
        print(f"总共记录了 {len(events_df)} 个事件")
    else:
        print("未检测到任何夹爪事件")


def process_single_episode(file_path: str, config: GripperConfig, output_dir: Optional[Path] = None) -> Dict:
    """处理单个episode文件，返回gripper事件数据"""
    # 获取文件名
    file_name = Path(file_path).stem
    
    # 从文件名提取episode_index (假设格式为episode_XXXXXX)
    try:
        episode_index = int(file_name.split('_')[-1])
    except:
        episode_index = 0
    
    print(f"\n{'='*60}")
    print(f"处理: {file_name} (Episode {episode_index})")
    print(f"{'='*60}")
    
    # 加载数据
    df = load_single_episode(file_path)
    
    # 提取夹爪状态
    left_gripper, right_gripper = extract_gripper_states(df, config)
    
    # 检测状态变化
    left_info = detect_gripper_changes(left_gripper, config)
    right_info = detect_gripper_changes(right_gripper, config)
    
    # 打印统计信息
    analyze_gripper_statistics(left_info, "左手夹爪", config)
    analyze_gripper_statistics(right_info, "右手夹爪", config)
    
    # 设置输出路径
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化
    plot_path = output_dir / f"{file_name}_gripper_analysis.png"
    plot_gripper_states(df, left_info, right_info, config, str(plot_path))
    
    # 合并并收集gripper事件数据
    print("\n处理夹爪事件数据...")
    gripper_changes = merge_gripper_events(left_info, right_info, frame_threshold=3)
    
    return {
        "episode_index": episode_index,
        "gripper_change": gripper_changes
    }


def merge_gripper_events(left_info: Dict, right_info: Dict, frame_threshold: int = 3) -> List[Dict]:
    """
    合并左右手夹爪事件
    如果左右手的同类事件（都是open或都是close）发生在相差不超过frame_threshold帧的时间内，
    则合并为一个事件
    
    Args:
        left_info: 左手夹爪信息
        right_info: 右手夹爪信息
        frame_threshold: 帧差阈值，默认3帧
    
    Returns:
        合并后的事件列表，按gripper_step排序
    """
    # 收集所有事件
    all_events = []
    
    # 添加左手事件
    for op in left_info['open_points']:
        all_events.append({
            "gripper_step": int(op),
            "gripper": "left",
            "event": "open"
        })
    
    for cp in left_info['close_points']:
        all_events.append({
            "gripper_step": int(cp),
            "gripper": "left",
            "event": "close"
        })
    
    # 添加右手事件
    for op in right_info['open_points']:
        all_events.append({
            "gripper_step": int(op),
            "gripper": "right",
            "event": "open"
        })
    
    for cp in right_info['close_points']:
        all_events.append({
            "gripper_step": int(cp),
            "gripper": "right",
            "event": "close"
        })
    
    # 按gripper_step排序
    all_events.sort(key=lambda x: x['gripper_step'])
    
    # 合并相近的同类事件
    merged_events = []
    used_indices = set()
    
    for i, event in enumerate(all_events):
        if i in used_indices:
            continue
        
        current_step = event['gripper_step']
        current_event_type = event['event']
        current_gripper = event['gripper']
        
        # 查找是否有相近的同类事件
        matched_idx = None
        
        for j in range(i + 1, len(all_events)):
            if j in used_indices:
                continue
            
            other_event = all_events[j]
            step_diff = abs(other_event['gripper_step'] - current_step)
            
            # 如果帧差超过阈值，后面的事件都不需要检查了
            if step_diff > frame_threshold:
                break
            
            # 检查是否是同类事件且是另一只手
            if (other_event['event'] == current_event_type and 
                other_event['gripper'] != current_gripper):
                matched_idx = j
                break
        
        # 如果找到匹配的事件，合并
        if matched_idx is not None:
            # 使用较小的gripper_step作为合并后的时间点
            merged_step = min(current_step, all_events[matched_idx]['gripper_step'])
            
            merged_events.append({
                "gripper_step": merged_step,
                "gripper": "left, right",  # 合并左右手
                "event": current_event_type
            })
            used_indices.add(i)
            used_indices.add(matched_idx)
        else:
            # 没有匹配的事件，单独保存
            merged_events.append({
                "gripper_step": current_step,
                "gripper": current_gripper,
                "event": current_event_type
            })
            used_indices.add(i)
    
    # 按gripper_step重新排序
    merged_events.sort(key=lambda x: x['gripper_step'])
    
    return merged_events


def main():
    # 配置夹爪参数
    config = GripperConfig(
        left_gripper_index=LEFT_GRIPPER_INDEX,      # 左手夹爪在observation.state中的索引
        right_gripper_index=RIGHT_GRIPPER_INDEX,    # 右手夹爪在observation.state中的索引
        one_means_open=True,                        # True: 1=打开,0=闭合; False: 1=闭合,0=打开
        value_range=(0.0, 1.0),                     # 值的范围
        open_threshold=OPEN_THRESHOLD,              # 打开阈值
        close_threshold=CLOSE_THRESHOLD,            # 闭合阈值
        min_duration=MIN_DURATION                   # 最小持续帧数
    )

    config.print_config()

    data_dir = DATA_PATH
    output_dir = GRIPPER_ANALYSIS_PATH
    
    print(f"\n{'='*60}")
    print(f"扫描目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")
    
    # 获取所有parquet文件
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.glob("*.parquet"))
    
    if len(parquet_files) == 0:
        print(f"\n错误: 在 {data_dir} 中未找到任何 .parquet 文件")
        return
    
    print(f"\n找到 {len(parquet_files)} 个 parquet 文件")
    
    # 收集所有episode的gripper事件数据
    all_episodes_data = []
    
    # 处理每个文件
    for i, file_path in enumerate(parquet_files, 1):
        print(f"\n进度: {i}/{len(parquet_files)}")
        try:
            episode_data = process_single_episode(str(file_path), config, output_dir)
            all_episodes_data.append(episode_data)
        except Exception as e:
            print(f"\n错误: 处理 {file_path.name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存所有数据到jsonl文件
    jsonl_path = Path(output_dir) / "gripper_events_all.jsonl"
    print(f"\n保存所有gripper事件到: {jsonl_path}")
    
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for episode_data in all_episodes_data:
            json_line = json.dumps(episode_data, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"成功保存 {len(all_episodes_data)} 个episode的gripper事件数据")
    
    print("\n" + "="*60)
    print(f"批量分析完成! 共处理 {len(parquet_files)} 个文件")
    print(f"结果保存在: {output_dir}")
    print(f"JSONL文件: {jsonl_path}")
    print("="*60)


if __name__ == "__main__":
    main()

