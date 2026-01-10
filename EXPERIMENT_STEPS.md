# LatentMAS + LoRA 完整实验步骤

## ⚠️ 重要更新 (2026-01-10)

### 代码改进说明

本项目已完成以下关键改进，确保LoRA训练能真正学习到关系抽取能力：

**1. 训练格式改进 (finetune_lora.py)**
- DocRED训练prompt现在包含完整的entity_list（实体列表）
- 明确说明常用关系类型（P17, P131, P27等）及其含义
- 输出格式说明与推理时完全一致
- 新增evidence字段训练（句子索引列表）

**2. 推理流程优化 (methods/latent_mas.py)**
- LoRA模型使用直接推理模式，跳过planner/critic/refiner/judger多agent流程
- 推理prompt与训练prompt完全一致，确保格式匹配
- 新增 `build_lora_extraction_prompt()` 专用函数

**3. 评估改进 (evaluate_extraction.py)**
- 智能JSON提取：支持从各种格式的模型输出中提取JSON
- 实体名称模糊匹配：忽略大小写差异
- 分关系类型统计：输出每个P-ID的Precision/Recall/F1
- 更详细的错误分析

**4. 数据格式简化 (data.py)**
- gold输出只包含`{"relations": [...]}`，不再包含raw_labels
- 保留raw_labels用于官方评估格式转换

---

## 前置准备

### 1. 检查环境
```powershell
# 检查CUDA和GPU
nvidia-smi

# 检查Python版本（需要3.8+）
python --version

# 检查依赖包
pip list | Select-String "torch|transformers|peft"
```

### 2. 安装依赖（如果还没安装）
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.50.0
pip install peft==0.14.0
pip install accelerate bitsandbytes
pip install qwen-vl-utils
pip install pillow opencv-python
```

### 3. 准备数据集

**FUNSD（表单理解）**
```
data/funsd/
  ├── instances_train.json    # COCO格式训练数据
  ├── test.json               # 测试数据
  ├── annotations/            # 原始标注
  └── imgs/                   # 图像文件
```

**DocRED（关系抽取）**
```
data/
  ├── test_docred.json        # 测试数据
  └── docred/
      └── train_annotated.json  # 训练数据
```

---

## 实验一：FUNSD基线测试（无LoRA）

### 步骤1：运行基线模型
```powershell
# 使用50个样本快速测试
python run.py `
    --task funsd `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --method latent_mas `
    --architecture hierarchical `
    --doc_path data/test.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --max_samples 50 `
    --batch_size 1 `
    --output_path results/funsd_baseline_50samples.json

# 记录结果：Entity F1, Relation F1, 输出格式正确率
```

**预期结果：**
- Entity F1: ~30-40%
- Relation F1: ~10-20%
- 常见问题：输出格式不规范、关系识别不准确

---

## 实验二：FUNSD LoRA微调（小规模快速验证）

### 步骤1：使用100个样本训练LoRA
```powershell
python finetune_lora.py `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --task funsd `
    --train_data data/funsd/instances_train.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --output_dir lora_weights/funsd_100samples `
    --max_train_samples 100 `
    --epochs 5 `
    --batch_size 2 `
    --gradient_accumulation_steps 8 `
    --learning_rate 2e-4 `
    --lora_r 16 `
    --lora_alpha 32

# 预计用时：30-60分钟（取决于GPU）
```

### 步骤2：测试LoRA模型
```powershell
python run.py `
    --task funsd `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --lora_weights lora_weights/funsd_100samples `
    --method latent_mas `
    --architecture hierarchical `
    --doc_path data/test.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --max_samples 50 `
    --batch_size 1 `
    --output_path results/funsd_lora100_50samples.json
```

### 步骤3：对比结果
```powershell
# 对比baseline和LoRA结果
python evaluate_extraction.py `
    --pred_file results/funsd_baseline_50samples.json `
    --gold_file data/test.json

python evaluate_extraction.py `
    --pred_file results/funsd_lora100_50samples.json `
    --gold_file data/test.json

# 预期提升：Entity F1 +5-10%, Relation F1 +10-15%
```

---

## 实验三：FUNSD完整训练（全部数据）

### 步骤1：使用全部训练数据微调
```powershell
# 使用4张GPU并行训练（如果只有1张GPU，去掉CUDA_VISIBLE_DEVICES）
$env:CUDA_VISIBLE_DEVICES="0,1,2,3"
python finetune_lora.py `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --task funsd `
    --train_data data/funsd/instances_train.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --output_dir lora_weights/funsd_full `
    --epochs 3 `
    --batch_size 2 `
    --gradient_accumulation_steps 8 `
    --learning_rate 2e-4

# 预计用时：2-4小时
```

### 步骤2：完整测试集评估
```powershell
python run.py `
    --task funsd `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --lora_weights lora_weights/funsd_full `
    --method latent_mas `
    --architecture hierarchical `
    --doc_path data/test.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --output_path results/funsd_lora_full.json

# 评估
python evaluate_extraction.py `
    --pred_file results/funsd_lora_full.json `
    --gold_file data/test.json
```

---

## 实验四：DocRED关系抽取（文本模型）

### 步骤1：基线测试
```powershell
python run.py `
    --task docred `
    --model_name Qwen/Qwen2.5-7B-Instruct `
    --method latent_mas `
    --architecture hierarchical `
    --doc_path data/test_docred.json `
    --max_samples 50 `
    --output_path results/docred_baseline_50samples.json
```

### 步骤2：LoRA微调（使用500个样本）
```powershell
python finetune_lora.py `
    --model_name Qwen/Qwen2.5-7B-Instruct `
    --task docred `
    --train_data data/docred/train_annotated.json `
    --output_dir lora_weights/docred_500samples `
    --max_train_samples 500 `
    --epochs 5 `
    --batch_size 4 `
    --gradient_accumulation_steps 4 `
    --learning_rate 1e-4

# 预计用时：1-2小时
```

### 步骤3：测试LoRA效果
```powershell
python run.py `
    --task docred `
    --model_name Qwen/Qwen2.5-7B-Instruct `
    --lora_weights lora_weights/docred_500samples `
    --method latent_mas `
    --architecture hierarchical `
    --doc_path data/test_docred.json `
    --max_samples 50 `
    --output_path results/docred_lora500_50samples.json

# 评估
python evaluate_extraction.py `
    --pred_file results/docred_lora500_50samples.json `
    --gold_file data/test_docred.json
```

---

## 实验五：消融实验（数据量对比）

### 测试不同训练样本数的影响

```powershell
# 100个样本
python finetune_lora.py --task funsd --max_train_samples 100 --output_dir lora_weights/funsd_100 --epochs 5 ...

# 300个样本
python finetune_lora.py --task funsd --max_train_samples 300 --output_dir lora_weights/funsd_300 --epochs 4 ...

# 500个样本
python finetune_lora.py --task funsd --max_train_samples 500 --output_dir lora_weights/funsd_500 --epochs 3 ...

# 全部样本
python finetune_lora.py --task funsd --output_dir lora_weights/funsd_full --epochs 3 ...

# 分别测试和对比结果
```

---

## 实验六：架构对比（Sequential vs Hierarchical）

```powershell
# Sequential架构 + LoRA
python run.py `
    --task funsd `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --lora_weights lora_weights/funsd_full `
    --method latent_mas `
    --architecture sequential `
    --doc_path data/test.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --output_path results/funsd_lora_sequential.json

# Hierarchical架构 + LoRA
python run.py `
    --task funsd `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --lora_weights lora_weights/funsd_full `
    --method latent_mas `
    --architecture hierarchical `
    --doc_path data/test.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --output_path results/funsd_lora_hierarchical.json

# 对比哪种架构效果更好
```

---

## 实验七：超参数调优

### 测试不同LoRA配置

```powershell
# 配置1：小rank（r=8）
python finetune_lora.py --task funsd --lora_r 8 --lora_alpha 16 --output_dir lora_weights/funsd_r8 ...

# 配置2：中rank（r=16，默认）
python finetune_lora.py --task funsd --lora_r 16 --lora_alpha 32 --output_dir lora_weights/funsd_r16 ...

# 配置3：大rank（r=32）
python finetune_lora.py --task funsd --lora_r 32 --lora_alpha 64 --output_dir lora_weights/funsd_r32 ...

# 对比：参数量 vs 性能 vs 训练时间
```

---

## 快速诊断命令

### 检查训练日志
```powershell
# 实时监控训练
Get-Content lora_weights/funsd_full/trainer_log.txt -Wait

# 检查最后几行
Get-Content lora_weights/funsd_full/trainer_log.txt -Tail 20
```

### 检查LoRA权重是否保存
```powershell
ls lora_weights/funsd_full/
# 应该看到：adapter_config.json, adapter_model.safetensors
```

### 快速验证单个样本
```powershell
python run.py `
    --task funsd `
    --model_name Qwen/Qwen3-VL-4B-Instruct `
    --lora_weights lora_weights/funsd_100samples `
    --doc_path data/test.json `
    --annotations_dir data/funsd/annotations `
    --image_dir data/funsd/imgs `
    --max_samples 1 `
    --output_path test_single.json
```

---

## 结果记录模板

创建 `experiment_results.txt` 记录每次实验：

```
实验日期：2026-01-09
任务：FUNSD
模型：Qwen3-VL-4B-Instruct
配置：
  - 训练样本：100
  - Epochs：5
  - Batch size：2
  - LoRA r：16
  - Learning rate：2e-4

结果：
  - Entity F1：45.2%
  - Relation F1：32.8%
  - 训练时间：45分钟
  - GPU显存：18GB

对比基线：
  - Entity F1提升：+8.5%
  - Relation F1提升：+15.3%

备注：格式输出明显改善，关系识别更准确
```

---

## 常见问题排查

### 问题1：CUDA out of memory
```powershell
# 解决方案：减小batch size或使用梯度累积
--batch_size 1 --gradient_accumulation_steps 16
```

### 问题2：图像加载失败
```powershell
# 检查图像路径
ls data/funsd/imgs/

# 确认instances_train.json中的file_name字段正确
```

### 问题3：训练loss不下降
```powershell
# 尝试调整学习率
--learning_rate 1e-4  # 降低学习率
--learning_rate 5e-4  # 提高学习率
```

### 问题4：推理时找不到LoRA权重
```powershell
# 检查文件是否存在
ls lora_weights/funsd_full/adapter_model.safetensors

# 确认路径正确
--lora_weights lora_weights/funsd_full  # 不要包含文件名
```

### 问题5：bf16不支持错误
**错误信息**：`ValueError: Your setup doesn't support bf16/gpu`

**原因**：GPU不支持bf16（需要Ampere架构或更新，如RTX 30系列、A100等）

**解决方案**：代码已自动检测GPU能力，会自动降级到fp16
- Ampere及以上（RTX 30/40系列、A100/H100）：使用bf16
- 较老GPU（RTX 20系列、V100、GTX 1080等）：自动使用fp16

**手动检查GPU能力**：
```python
import torch
print(torch.cuda.get_device_capability())  # (8, 0)及以上支持bf16
```

### 问题6：训练时没有梯度（Qwen3模型）
**错误信息**：`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

**原因1**：Qwen3模型的层命名可能与硬编码的`target_modules`不匹配，导致LoRA没有应用到任何层

**原因2**：`gradient_checkpointing=True` 与PEFT LoRA在某些配置下不兼容

**解决方案**：
1. 代码已更新为**自动检测模型的Linear层**，适配所有Qwen系列模型
2. 禁用gradient_checkpointing（虽然会增加显存使用，但保证训练稳定）
3. 显式启用LoRA参数的梯度

**验证LoRA是否正确应用**：
训练开始时会打印：
```
LoRA target modules: ['q_proj', 'k_proj', 'v_proj', ...]
trainable params: 43,646,976 || all params: 8,234,382,336 || trainable%: 0.5301
```
确保trainable%大于0！

**如果显存不足**：
```powershell
# 减小batch size并增加梯度累积
--batch_size 1 --gradient_accumulation_steps 16

# 或减小LoRA rank
--lora_r 8 --lora_alpha 16
```

---

## 推荐实验流程（新手）

1. **Day 1：环境准备** ✅
   - 安装依赖
   - 准备数据
   - 运行基线（10个样本快速测试）

2. **Day 2：快速验证** 🚀
   - 训练100样本LoRA（1小时）
   - 测试效果
   - 调试问题

3. **Day 3：完整训练** 🎯
   - 训练全部数据LoRA（3-4小时）
   - 完整评估
   - 记录结果

4. **Day 4：消融实验** 🔬
   - 对比不同数据量
   - 对比不同架构
   - 撰写报告

---

## 预期性能指标

| 任务 | 基线F1 | LoRA F1 | 提升 |
|------|--------|---------|------|
| FUNSD Entity | 35% | 50-60% | +15-25% |
| FUNSD Relation | 15% | 35-45% | +20-30% |
| DocRED | 25% | 40-50% | +15-25% |
| CORD | 40% | 60-70% | +20-30% |
| FinER | 45% | 65-75% | +20-30% |

*实际结果取决于数据质量、模型配置和训练参数*
