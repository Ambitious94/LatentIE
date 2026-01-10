"""
LoRA微调脚本 - 针对文档信息抽取任务微调Qwen-VL模型

使用方法:
python finetune_lora.py \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --task funsd \
    --train_data /data/funsd/instances_train.json \
    --annotations_dir /data/funsd/annotations \
    --image_dir /data/funsd/imgs \
    --output_dir ./lora_weights/funsd \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
"""

import os
import json
import torch
import argparse
from typing import List, Dict
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from PIL import Image
from data import load_funsd, load_docred, load_cord, load_finer


class DocumentExtractionDataset(Dataset):
    """文档信息抽取数据集"""
    
    def __init__(self, data_items: List[Dict], processor, task: str):
        self.data_items = data_items
        self.processor = processor
        self.task = task
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        # 构建输入消息
        question = item.get("question", "")
        gold = item.get("gold", "{}")
        image = item.get("image")
        
        # 构建指令格式（与推理时的LoRA prompts一致）
        if self.task == "funsd":
            instruction = "Task: Extract form fields and relations"
        elif self.task == "docred":
            instruction = "Task: Extract document relations"
        elif self.task == "cord":
            instruction = "Task: Extract receipt information"
        elif self.task == "finer":
            instruction = "Task: Extract financial entities"
        else:
            instruction = "Task: Extract information"
        
        # 构建消息格式（简洁版，模型会学习输出格式）
        if image:
            messages = [
                {"role": "system", "content": "You are a document extraction specialist. Output valid JSON only."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{instruction}\n\nDocument:\n{question}\n\nOutput complete JSON:"}
                ]},
                {"role": "assistant", "content": gold}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a document extraction specialist. Output valid JSON only."},
                {"role": "user", "content": f"{instruction}\n\nDocument:\n{question}\n\nOutput complete JSON:"},
                {"role": "assistant", "content": gold}
            ]
        
        # 使用processor处理
        if hasattr(self.processor, 'apply_chat_template'):
            # Qwen系列模型支持chat template
            text = self.processor.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback：手动构建文本
            text = ""
            for msg in messages:
                if isinstance(msg["content"], list):
                    text += "\n".join([c["text"] for c in msg["content"] if c.get("type") == "text"])
                else:
                    text += msg["content"]
                text += "\n"
        
        if image:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=2048,
                truncation=True
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=2048,
                truncation=True
            )
        
        # 创建labels (mask掉input部分，只计算output的loss)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # 简单实现：所有tokens都参与训练
        # 更好的实现：只计算assistant回复部分的loss
        labels = input_ids.clone()
        # 获取pad_token_id并mask掉padding部分
        pad_token_id = self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else self.processor.pad_token_id
        labels[labels == pad_token_id] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        # 只有VL模型才有pixel_values
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result


def load_training_data(args):
    """加载训练数据"""
    print(f"Loading training data for {args.task}...")
    
    if args.task == "funsd":
        data_iter = load_funsd(
            doc_path=args.train_data,
            split="train",
            mode="full",
            annotations_dir=args.annotations_dir,
            images_dir=args.image_dir
        )
    elif args.task == "docred":
        data_iter = load_docred(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    elif args.task == "cord":
        data_iter = load_cord(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    elif args.task == "finer":
        data_iter = load_finer(
            doc_path=args.train_data,
            split="train",
            mode="full"
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")
    
    data_items = list(data_iter)
    print(f"Loaded {len(data_items)} training samples")
    
    return data_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--task", type=str, required=True, choices=["funsd", "docred", "cord", "finer"])
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Maximum number of training samples (use all if not specified)")
    parser.add_argument("--use_vision_model", action="store_true", help="Use vision-language model (auto-detect if not specified)")
    args = parser.parse_args()
    
    # 自动检测是否应该使用VL模型
    if not args.use_vision_model:
        # FUNSD和CORD默认使用VL模型，DocRED和FinER使用文本模型
        if args.task in ["funsd", "cord"]:
            args.use_vision_model = True
            print(f"[Auto] Task {args.task} detected, using vision-language model")
        else:
            args.use_vision_model = False
            print(f"[Auto] Task {args.task} detected, using text-only model")
    
    # 加载模型和processor
    print(f"Loading model: {args.model_name}")
    
    if args.use_vision_model:
        # 加载VL模型
        from transformers import AutoProcessor
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_name)
    else:
        # 加载纯文本模型
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoTokenizer.from_pretrained(args.model_name)
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
            processor.pad_token_id = processor.eos_token_id
    
    # 配置LoRA
    print("Configuring LoRA...")
    
    # 自动查找模型中的linear层作为target_modules
    # 这样可以适配不同模型架构（Qwen2, Qwen3, Qwen-VL等）
    import re
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # 提取层名称的最后一部分（如 q_proj, k_proj等）
            layer_name = name.split('.')[-1]
            if layer_name not in target_modules and not layer_name.startswith('lm_head'):
                target_modules.append(layer_name)
    
    # 如果自动检测失败，使用常见的target_modules
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    print(f"LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 确保模型在训练模式并启用梯度
    model.train()
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    # 加载数据
    train_data = load_training_data(args)
    
    # 限制训练样本数量（如果指定）
    if args.max_train_samples is not None and args.max_train_samples > 0:
        original_size = len(train_data)
        train_data = train_data[:args.max_train_samples]
        print(f"Using {len(train_data)} out of {original_size} training samples")
    else:
        print(f"Using all {len(train_data)} training samples")
    
    train_dataset = DocumentExtractionDataset(train_data, processor, args.task)
    
    # 检测GPU是否支持bf16
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if use_bf16:
        print("Using bf16 training (GPU supports it)")
    else:
        print("Using fp16 training (bf16 not supported, falling back to fp16)")
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=False,  # 禁用以避免与LoRA冲突
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: {k: torch.stack([d[k] for d in x if d[k] is not None]) if x[0][k] is not None else None for k in x[0].keys()}
    )
    
    print("Starting training...")
    trainer.train()
    
    # 保存
    print(f"Saving LoRA weights to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
