# 📋 文档抽取实验详细步骤

## 1. 环境准备

### 1.1 创建虚拟环境
```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas
```

### 1.2 安装依赖
```bash
pip install -r requirements.txt
```

### 1.3 设置HuggingFace缓存目录（可选）
```bash
# Windows PowerShell
$env:HF_HOME = "E:\huggingface_cache"
$env:TRANSFORMERS_CACHE = $env:HF_HOME

# Linux/Mac
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
```

---

## 2. 数据准备

### 2.1 DocRED（文档级关系抽取）
```
数据位置: e:/Edge Download/dev.json  (有标签，可评估)
          e:/Edge Download/test (1).json  (无标签，需提交CodaLab)
格式: {"title": "", "sents": [[]], "vertexSet": [...], "labels": [...]}
```

### 2.2 CORD（收据信息抽取）
```
数据位置: e:/Edge Download/samples.json
格式: {"samples": [{"filepath": "data/xxx.png", "num_items": 22, "total_price": "..."}]}
需要: samples.json同目录下有data/文件夹包含图片
```

### 2.3 FUNSD（表单理解）
```
数据位置: instances_test.json
格式: COCO-style {"images": [...], "annotations": [...]}
需要: annotations/文件夹（包含segm_file）和images/文件夹
```

### 2.4 FinER-139（金融实体识别）
```
格式: {"tokens": [...], "ner_tags": [...]}
需要: 同目录下有tag2id.json映射文件
```

---

## 3. 运行实验

### 3.1 DocRED 关系抽取

#### Sequential模式（latent_mas）
```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task docred \
    --doc_path "e:/Edge Download/dev.json" \
    --prompt sequential \
    --extraction_mode chunks \
    --chunk_size 3000 \
    --max_samples 100 \
    --output_path docred_latent_sequential.json
```

#### Hierarchical模式
```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task docred \
    --doc_path "e:/Edge Download/dev.json" \
    --prompt hierarchical \
    --extraction_mode partitioned \
    --num_partitions 3 \
    --max_samples 100 \
    --output_path docred_latent_hierarchical.json
```

#### TextMAS对比实验
```bash
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task docred \
    --doc_path "e:/Edge Download/dev.json" \
    --prompt sequential \
    --max_samples 100 \
    --output_path docred_textmas.json
```

#### Baseline对比
```bash
python run.py \
    --method baseline \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --task docred \
    --doc_path "e:/Edge Download/dev.json" \
    --max_samples 100 \
    --output_path docred_baseline.json
```

---

### 3.2 CORD 收据抽取（多模态）

#### 使用视觉模型
```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2-VL-2B-Instruct \
    --task cord \
    --doc_path "e:/Edge Download/samples.json" \
    --use_vision_model \
    --prompt sequential \
    --max_samples 50 \
    --output_path cord_results.json
```

#### 使用更大的模型
```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --task cord \
    --doc_path "e:/Edge Download/samples.json" \
    --use_vision_model \
    --max_samples 50 \
    --output_path cord_7b_results.json
```

---

### 3.3 FUNSD 表单理解（多模态）

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2-VL-7B-Instruct \
    --task funsd \
    --doc_path "path/to/instances_test.json" \
    --use_vision_model \
    --prompt sequential \
    --max_samples 50 \
    --output_path funsd_results.json
```

---

### 3.4 FinER-139 金融实体识别

```bash
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen2.5-4B-Instruct \
    --task finer \
    --doc_path "path/to/finer_data.json" \
    --prompt sequential \
    --max_samples 100 \
    --output_path finer_results.json
```

---

## 4. 关键参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--method` | 方法选择: baseline / text_mas / latent_mas | 必填 |
| `--model_name` | 模型名称 | 必填 |
| `--task` | 任务: docred / cord / funsd / finer | gsm8k |
| `--doc_path` | 数据文件路径 | None |
| `--prompt` | MAS架构: sequential / hierarchical | sequential |
| `--extraction_mode` | 处理模式: chunks / partitioned | chunks |
| `--chunk_size` | 每块字符数 | 3000 |
| `--chunk_overlap` | 块之间重叠 | 300 |
| `--num_partitions` | 分区数量(hierarchical) | 3 |
| `--max_samples` | 最大样本数，-1为全部 | -1 |
| `--max_new_tokens` | 最大生成token数 | 4096 |
| `--latent_steps` | LatentMAS潜空间步数 | 0 |
| `--use_vision_model` | 使用视觉模型 | False |
| `--output_path` | 结果保存路径 | 自动生成 |

---

## 5. 查看结果

### 5.1 终端输出
运行过程中会实时显示：
- 每个样本的Agent处理过程
- 预测结果 vs 标准答案
- 最终评估指标

### 5.2 结果文件格式
```json
{
  "summary": {
    "method": "latent_mas",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "task": "docred",
    "precision": 0.6234,
    "recall": 0.5821,
    "f1": 0.6021,
    "total_time_sec": 3456.78
  },
  "predictions": [
    {
      "question": "文档内容...",
      "prediction": "{\"relations\": [...]}",
      "gold": "{\"relations\": [...]}",
      "correct": true,
      "agents": [...]
    }
  ]
}
```

---

## 6. 评估指标

### DocRED
- Precision, Recall, F1（关系三元组级别）

### CORD
- 逐字段准确率：num_items, subtotal_price, total_price等
- Overall Accuracy

### FUNSD
- Entity F1（实体识别）
- Relation F1（关系抽取）
- Overall F1

### FinER-139
- Precision, Recall, F1（实体级别）

---

## 7. 推荐实验方案

### 7.1 快速验证（小规模）
```bash
# 先用少量样本测试流程是否正确
python run.py --method latent_mas --model_name Qwen/Qwen2.5-1.5B-Instruct \
              --task docred --doc_path "dev.json" --max_samples 10
```

### 7.2 完整对比实验
```bash
# 对每个数据集，运行三种方法：
# 1. baseline (单Agent)
# 2. text_mas (文本传递多Agent)
# 3. latent_mas (潜空间多Agent)

# 并对比两种架构：
# - sequential (顺序)
# - hierarchical (层级)
```

### 7.3 模型规模对比
```bash
# 对比不同模型大小的效果：
# - Qwen2.5-1.5B-Instruct
# - Qwen2.5-4B-Instruct
# - Qwen2.5-7B-Instruct
# - Qwen2.5-14B-Instruct
```

---

## 8. 常见问题

### Q: 显存不足
```bash
# 使用较小的模型
--model_name Qwen/Qwen2.5-1.5B-Instruct

# 减少batch size
--generate_bs 1

# 减少最大生成长度
--max_new_tokens 2048
```

### Q: 速度太慢
```bash
# 使用vLLM加速（需要额外安装）
pip install vllm
python run.py ... --use_vllm
```

### Q: CUDA错误
```bash
# 指定GPU设备
--device cuda:0
# 或使用CPU
--device cpu
```

---

## 9. 输出格式要求（提交官方评测）

### DocRED CodaLab提交格式
```json
[
  {"title": "文档标题", "h_idx": 0, "t_idx": 1, "r": "P17"},
  ...
]
```

如需转换，可手动处理results.json中的predictions。
