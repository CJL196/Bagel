# BAGEL 模型微调详细操作指南

## 1. 环境准备

### 1.1 硬件要求
- GPU: 推荐使用 A100/H100 等大显存显卡
- 内存: 至少 32GB RAM
- 存储: 足够空间存放数据集和模型检查点

### 1.2 环境变量设置
```bash
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整
```

## 2. 数据准备

### 2.1 下载示例数据集
```bash
# 下载官方提供的示例数据集
wget -O bagel_example.zip \
  https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
  
# 解压到数据目录
unzip bagel_example.zip -d /data
```

### 2.2 数据集目录结构
解压后的数据集应具有以下目录结构：
```
bagel_example/
├── t2i/                           # 文本生图数据（parquet格式）
├── editing/                       # 图像编辑数据（parquet格式）
│   ├── seedxedit_multi/
│   └── parquet_info/
└── vlm/                          # 视觉语言模型数据
    ├── images/                    # 图像文件（JPEG/PNG）
    └── llava_ov_si.jsonl          # 对话数据
```

### 2.3 配置数据路径
编辑 `data/dataset_info.py` 文件，更新所有 `your_data_path` 占位符为实际数据路径：

```python
DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': '/data/bagel_example/t2i',  # 更新为实际路径
            'num_files': 10,
            'num_total_samples': 1000,
        },
    },
    'unified_edit': {
        'seedxedit_multi': {
            'data_dir': '/data/bagel_example/editing/seedxedit_multi',  # 更新路径
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": '/data/bagel_example/editing/parquet_info/seedxedit_multi_nas.json',
        },
    },
    'vlm_sft': {
        'llava_ov': {
            'data_dir': '/data/bagel_example/vlm/images',  # 更新路径
            'jsonl_path': '/data/bagel_example/vlm/llava_ov_si.jsonl',
            'num_total_samples': 1000
        },
    },
}
```

### 2.4 检查数据配置
确认 `data/configs/example.yaml` 中的配置符合需求。该文件定义了不同任务的数据采样权重和预处理参数。

## 3. 模型准备

### 3.1 下载预训练模型
从 HuggingFace 下载 BAGEL 预训练模型：
```bash
# 方法1: 使用 git lfs
git lfs clone https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT models/BAGEL-7B-MoT

# 方法2: 使用 huggingface-hub
pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('ByteDance-Seed/BAGEL-7B-MoT', local_dir='models/BAGEL-7B-MoT')"
```

### 3.2 模型目录结构
确保模型目录包含以下文件：
```
models/BAGEL-7B-MoT/
├── llm_config.json          # 语言模型配置
├── vit_config.json          # 视觉模型配置
├── ae.safetensors           # VAE模型权重
├── ema.safetensors          # EMA模型权重
├── tokenizer.json           # 分词器
└── ... (其他模型文件)
```

## 4. 微调训练

### 4.1 单GPU微调命令
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --num_shard 1 \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --results_dir ./results \
  --checkpoint_dir ./results/checkpoints \
  --total_steps 10000 \
  --save_every 1000 \
  --warmup_steps 500
```

### 4.2 多GPU微调命令
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --num_shard 8 \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 10 \
  --lr 2e-5 \
  --num_worker 4 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --results_dir ./results \
  --checkpoint_dir ./results/checkpoints \
  --total_steps 50000 \
  --save_every 2000 \
  --warmup_steps 2000
```

### 4.3 多节点训练命令（可选）
```bash
# 节点0（主节点）
PYTHONPATH=. torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=<主节点IP> \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path models/BAGEL-7B-MoT \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --lr 2e-5 \
  --num_worker 8 \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240

# 节点1（从节点）
PYTHONPATH=. torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=8 \
  --master_addr=<主节点IP> \
  --master_port=29500 \
  train/pretrain_unified_navit.py \
  [相同参数...]
```

## 5. 重要参数说明

### 5.1 模型参数
- `--model_path`: 预训练模型路径
- `--layer_module`: 解码器层类型，必须为 `Qwen2MoTDecoderLayer`
- `--max_latent_size`: 最大潜在尺寸，微调时必须设为64
- `--finetune_from_hf`: 从HuggingFace模型微调，设为True
- `--finetune-from-ema`: 从EMA权重开始微调

### 5.2 训练参数
- `--lr`: 学习率，建议2e-5
- `--total_steps`: 总训练步数
- `--warmup_steps`: 预热步数
- `--save_every`: 保存检查点频率
- `--auto_resume`: 自动恢复训练

### 5.3 数据参数
- `--expected_num_tokens`: 期望token数量
- `--max_num_tokens`: 最大token数量
- `--max_num_tokens_per_sample`: 每个样本最大token数
- `--num_worker`: 数据加载工作进程数

### 5.4 任务特化参数
```bash
# 仅文本生图任务
--visual_und=False

# 仅视觉理解任务
--visual_gen=False

# 同时训练两个任务（默认）
--visual_gen=True --visual_und=True
```

## 6. 监控和调试

### 6.1 训练日志
训练过程中会输出类似以下的日志：
```
[2025-01-25 10:00:00] (step=0000000) Train Loss mse: 0.4063, Train Loss ce: 0.5504, Train Steps/Sec: 0.01
[2025-01-25 10:00:03] (step=0000001) Train Loss mse: 0.4121, Train Loss ce: 0.8152, Train Steps/Sec: 0.44
[2025-01-25 10:00:06] (step=0000002) Train Loss mse: 0.3876, Train Loss ce: 1.3411, Train Steps/Sec: 0.40
```

### 6.2 Weights & Biases 监控
```bash
# 设置W&B API密钥
export WANDB_API_KEY=your_wandb_api_key

# 在训练命令中添加W&B参数
--wandb_project bagel_finetune \
--wandb_name my_experiment \
--wandb_offline False
```

### 6.3 常见问题解决
1. **内存不足**: 减少 `max_num_tokens` 和 `max_num_tokens_per_sample`
2. **数据加载错误**: 检查 `data/dataset_info.py` 中的路径配置
3. **模型加载失败**: 确认 `max_latent_size=64` 设置正确
4. **训练速度慢**: 调整 `num_worker` 和 `prefetch_factor`

## 7. 模型保存和推理

### 7.1 检查点位置
训练完成的模型将保存在：
```
./results/checkpoints/
├── step_1000/
├── step_2000/
├── ...
└── step_final/
```

### 7.2 模型推理
使用训练好的模型进行推理：
```python
from app import setup_models, inferencer

# 加载微调后的模型
model_path = "./results/checkpoints/step_final"
model, vae_model, tokenizer, new_token_ids = setup_models(model_path)

# 进行推理
result = inferencer.generate(prompt="生成一张猫的图片")
```

## 8. 性能优化建议

### 8.1 硬件优化
- 使用NVMe SSD存储数据集以提高I/O性能
- 确保充足的CPU和内存资源
- 使用多GPU并行训练

### 8.2 超参数调优
- 根据数据集大小调整学习率和训练步数
- 使用梯度累积来模拟更大的批次大小
- 适当调整warmup步数比例

### 8.3 数据优化
- 预处理数据以减少训练时的计算开销
- 使用合适的数据并行策略
- 定期清理缓存和临时文件

---

**注意事项**：
1. 微调时必须设置 `max_latent_size=64` 以正确加载预训练权重
2. `num_used_data` 的总和应大于 `NUM_GPUS × NUM_WORKERS`
3. 对于小数据集，建议使用 `num_worker=1`
4. 调试时可以使用更小的token限制参数