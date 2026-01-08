# AGD20K 评估脚本使用说明

## 文件: eval_agd20k.py

### 功能描述
该脚本用于从AGD20K数据集的Unseen测试集中生成SAM3模型的affordance热力图预测结果。

### 主要功能

#### 1. **single_image_process** - 处理单张图片
处理指定的affordance-object组合的单张图片。

**参数：**
- `processor`: Sam3Processor实例
- `affordance_label`: 行为标签，如 "hit"
- `object_name`: 物体名称，如 "axe"  
- `img_name`: 图片文件名，如 "axe_000001.jpg"

**示例：**
```python
single_image_process(processor, "hit", "axe", "axe_000001.jpg")
```

#### 2. **batch_process** - 批量处理某个对象的所有图片
处理指定affordance-object组合文件夹下的所有图片。

**参数：**
- `processor`: Sam3Processor实例
- `affordance_label`: 行为标签，如 "hit"
- `object_name`: 物体名称，如 "axe"

**示例：**
```python
batch_process(processor, "hit", "axe")
```

#### 3. **process_all_from_jsonl** - 处理jsonl中的所有对象
从jsonl文件读取所有affordance-object记录，自动批量处理全部数据。

**示例：**
```python
process_all_from_jsonl(processor)
```

### 配置路径
```python
JSONL_PATH = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/agd20k_unseen_testset.jsonl"
EGOCENTRIC_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/egocentric"
OUTPUT_DIR = "/home/nightbreeze/research/Data/AGD20K/Unseen/testset/unseen-zeroshot/result"
```

### 工作流程

1. **读取prompt**: 从jsonl文件中查找对应affordance_label和object_name的sam3_prompt
2. **加载图片**: 从egocentric目录读取测试图片
3. **模型推理**: 使用SAM3模型和prompt生成热力图
4. **后处理**:
   - 高斯模糊 (sigma=2.0)
   - 归一化到0-1范围
   - 转换为0-255灰度值
   - 调整大小到224x224
5. **保存**: 保存为PNG灰度图到result目录

### 输出格式
- 分辨率: 224x224
- 格式: PNG灰度图 (0-255)
- 已应用: 高斯模糊 + 归一化

### 目录结构
```
输入:
egocentric/{affordance_label}/{object_name}/*.jpg

输出:
result/{affordance_label}/{object_name}/*.png
```

### 运行示例

```bash
cd /home/nightbreeze/research/Code/AffVLA/affvla
source .venv/bin/activate
python src/sam3/tools/eval_agd20k.py
```

### 注意事项
1. jsonl文件中sam3_prompt为空的记录会被跳过
2. 热力图会自动创建输出目录
3. 所有错误会被捕获并打印，不会中断整个处理流程
