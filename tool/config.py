"""
统一配置文件 - 切换数据集时只需修改这个文件
"""

# ==================== 基础配置 ====================
DATASET_NAME = "robotwin"
TASK_NAME = "beat_block_hammer-aloha-agilex_clean_50"
NUM_SEGMENTS = 2
SAVE_IMAGE_KEY = "rgb_global"

# ==================== 特殊规则 ====================
SPATIAL_RULES = None
# SPATIAL_RULES = {
#     "check_indices": [0],
#     "check_strings": ["handle"],
#     "modify_strings": [" and the handle of the skillet"]
# }

# ==================== 夹爪配置 ====================
LEFT_GRIPPER_INDEX = 6
RIGHT_GRIPPER_INDEX = 13
OPEN_THRESHOLD = 0.85
CLOSE_THRESHOLD = 0.15
MIN_DURATION = 3

# ==================== API 配置 ====================
API_KEY = "ms-69cd41b2-f1ac-437d-9cd6-950f0e33d8c4"
API_BASE_URL = "https://api-inference.modelscope.cn/v1"
MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"
STREAM = False
TEMPERATURE = 0.1
API_SLEEP_TIME = 5

# ==================== 路径配置 ====================
BASE_PATH = f"/home/nightbreeze/research/Data/{DATASET_NAME}"
LEROBOT_PATH = f"{BASE_PATH}/lerobotV2.1_data/{TASK_NAME}"
DATA_PATH = f"{LEROBOT_PATH}/data/chunk-000"
VIDEO_PATH = f"{LEROBOT_PATH}/videos/chunk-000/{SAVE_IMAGE_KEY}"
IMAGE_PATH = f"{LEROBOT_PATH}/images/{SAVE_IMAGE_KEY}"
META_PATH = f"{LEROBOT_PATH}/meta"
GRIPPER_ANALYSIS_PATH = f"{BASE_PATH}/gripper_analysis_results/{TASK_NAME}"
