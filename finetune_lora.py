"""
LoRA微调脚本 - 针对文档信息抽取任务微调Qwen模型

**推荐训练模式**: direct (task-only)
- 训练: LoRA学习 "文档内容 → 标准JSON输出"
- 推理: 同一个LoRA可用于sequential、hierarchical或任何自定义LatentMAS模式
- 原理: LatentMAS框架在推理时通过不同prompts调用LoRA模型构建多agent协作

**高级模式**: latent_mas (agent-aware)  
- 训练: LoRA学习完整的 "Planner→Critic→Refiner→Judger" 4-agent交互流程
- 推理: 必须使用训练时相同的prompt_style
- 适用场景: 希望将特定协作模式"烧"进模型权重

使用方法:
# 推荐: Task-only训练
python finetune_lora.py \
    --model_name Qwen/Qwen3-VL-4B-Instruct \
    --task funsd \
    --train_data /data/funsd/instances_train.json \
    --output_dir ./lora_weights/funsd \
    --training_mode direct \
    --epochs 3

# 推理时可用于任何模式
python run.py --method latent_mas --prompt sequential --lora_weights ./lora_weights/funsd
python run.py --method latent_mas --prompt hierarchical --lora_weights ./lora_weights/funsd
"""

import os
import json
import torch
import argparse
from typing import List, Dict, Tuple
from transformers import AutoModelForVision2Seq, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from PIL import Image
from data import load_funsd, load_docred, load_cord, load_finer


# ============= Agent Prompts for LatentMAS Training =============

def get_agent_prompts(task: str, question: str, entity_list: str = "", mode: str = "sequential") -> Dict[str, str]:
    """
    获取4个Agent的prompt模板
    
    Args:
        task: 任务类型 (docred/funsd/cord/finer)
        question: 文档内容
        entity_list: 实体列表（DocRED专用）
        mode: sequential(顺序协作) 或 hierarchical(并行分工)
    """
    
    if mode == "hierarchical":
        return get_agent_prompts_hierarchical(task, question, entity_list)
    else:
        return get_agent_prompts_sequential(task, question, entity_list)


def get_agent_prompts_sequential(task: str, question: str, entity_list: str = "") -> Dict[str, str]:
    """Sequential模式: Planner→Critic→Refiner→Judger 顺序协作"""
    
    if task == "docred":
        planner = f"""You are a Document Scanner Agent (Phase 1: Information Discovery).

Task: Document-level relation extraction.

Document:
{question}

Entities in document:
{entity_list}

Instructions:
- Carefully scan and identify all entity mentions
- Note potential relationships between entities
- Consider sentence indices as evidence
- Store findings in latent format

Begin scanning and record findings:"""

        critic = f"""You are a Document Validator Agent (Phase 2: Cross-Verification).

Task: Verify relation extraction accuracy.

Document:
{question}

Instructions:
- Cross-check entity relationships against document
- Verify evidence sentence indices
- Identify missing or incorrect relations
- Note corrections in latent format

Continue verification:"""

        refiner = f"""You are a Document Structuring Agent (Phase 3: Organization).

Task: Organize extracted relations.

Instructions:
- Consolidate verified relations
- Resolve any conflicts
- Prepare final extraction structure
- Use Wikidata P-IDs for relations

Continue organization:"""

        judger = f"""Task: Output final relation extraction JSON.

Entities: {entity_list}

Common relations: P17(country), P131(located in), P27(citizenship), P569(birth date), P570(death date), P19(birthplace), P20(death place), P69(educated at), P108(employer), P40(child), P26(spouse).

Output JSON format:
{{"relations": [{{"head": "Entity", "relation": "P17", "tail": "Country", "evidence": [0]}}]}}

Based on previous analysis, output ONLY the final JSON:"""

    elif task == "funsd":
        planner = f"""You are a Form Scanner Agent (Phase 1: Field Detection).

Task: Form understanding and field extraction.

Form content:
{question}

Instructions:
- Identify all form fields and their types
- Classify as: question/answer/header/other
- Note spatial relationships
- Store findings in latent format

Begin scanning:"""

        critic = f"""You are a Form Validator Agent (Phase 2: Link Verification).

Task: Verify form field relationships.

Instructions:
- Cross-check question-answer pairings
- Verify field classifications
- Identify orphaned fields
- Note corrections in latent format

Continue verification:"""

        refiner = f"""You are a Form Structuring Agent (Phase 3: Relationship Mapping).

Task: Finalize form structure.

Instructions:
- Consolidate question-answer pairs
- Confirm all field labels
- Prepare final linking structure

Continue organization:"""

        judger = f"""Task: Output final form extraction JSON.

Format:
{{"entities": [{{"text": "...", "label": "question|answer|header|other"}}], "relations": [{{"head": "question text", "tail": "answer text"}}]}}

Based on previous analysis, output ONLY the final JSON:"""

    elif task == "cord":
        planner = f"""You are a Receipt Scanner Agent (Phase 1: Item Detection).

Task: Receipt/invoice information extraction.

Receipt content:
{question}

Instructions:
- Identify all menu items with prices
- Note subtotal, tax, service charges
- Locate total amount
- Store findings in latent format

Begin scanning:"""

        critic = f"""You are a Receipt Validator Agent (Phase 2: Amount Verification).

Task: Verify extracted amounts.

Instructions:
- Cross-check item prices and totals
- Verify mathematical consistency
- Identify missing amounts
- Note corrections in latent format

Continue verification:"""

        refiner = f"""You are a Receipt Structuring Agent (Phase 3: Total Calculation).

Task: Finalize receipt structure.

Instructions:
- Consolidate all items and amounts
- Confirm total calculations
- Prepare final structure

Continue organization:"""

        judger = f"""Task: Output final receipt extraction JSON.

Format:
{{"num_items": N, "subtotal_price": "X.XX", "service_price": "", "tax_price": "", "total_price": "X.XX", "etc": ""}}

Based on previous analysis, output ONLY the final JSON:"""

    elif task == "finer":
        planner = f"""You are a Financial Entity Scanner (Phase 1: Entity Detection).

Task: Financial named entity recognition.

Text:
{question}

Entity types: PER, ORG, LOC, MONEY, DATE, PERCENT, STOCK, METRIC, PRODUCT, LAW

Instructions:
- Identify all financial entities
- Note entity boundaries (start/end positions)
- Classify entity types
- Store findings in latent format

Begin scanning:"""

        critic = f"""You are a Financial Entity Validator (Phase 2: Type Verification).

Task: Verify entity classifications.

Instructions:
- Cross-check entity type assignments
- Verify character positions
- Identify overlapping or missing entities
- Note corrections in latent format

Continue verification:"""

        refiner = f"""You are a Financial Entity Structuring Agent (Phase 3: Position Mapping).

Task: Finalize entity boundaries.

Instructions:
- Consolidate entity spans
- Confirm type classifications
- Calculate exact positions

Continue organization:"""

        judger = f"""Task: Output final entity extraction JSON.

Format:
{{"entities": [{{"text": "Apple", "type": "ORG", "start": 0, "end": 5}}]}}

Based on previous analysis, output ONLY the final JSON:"""

    else:
        planner = f"Scan document:\n{question}"
        critic = "Verify findings."
        refiner = "Organize results."
        judger = "Output final JSON:"
    
    return {
        "planner": planner,
        "critic": critic,
        "refiner": refiner,
        "judger": judger
    }


def get_agent_prompts_hierarchical(task: str, question: str, entity_list: str = "") -> Dict[str, str]:
    """Hierarchical模式: 多个partition并行读取，最后汇总"""
    
    if task == "docred":
        planner = f"""You are Document Partition Reader 1 (Focus: Entity Detection).

Task: Document-level relation extraction.

Document:
{question}

Instructions:
- Extract entities and their mentions from first part
- Note potential relationships
- Store findings in latent format

Partition 1 extraction:"""

        critic = f"""You are Document Partition Reader 2 (Focus: Relation Verification).

Task: Verify and extract relations from middle part.

Document:
{question}

Instructions:
- Cross-check relationships in middle section
- Verify evidence sentences
- Store findings in latent format

Partition 2 extraction:"""

        refiner = f"""You are Document Partition Reader 3 (Focus: Evidence Collection).

Task: Extract evidence from final part.

Document:
{question}

Instructions:
- Identify supporting sentences
- Confirm entity-relation mappings
- Store findings in latent format

Partition 3 extraction:"""

        judger = f"""You are Integration Agent. Consolidate all partition findings.

Entities: {entity_list}

Common relations: P17(country), P131(located in), P27(citizenship), P569(birth date), P570(death date), P19(birthplace), P20(death place), P69(educated at), P108(employer), P40(child), P26(spouse).

Output JSON format:
{{"relations": [{{"head": "Entity", "relation": "P17", "tail": "Country", "evidence": [0]}}]}}

Based on all partitions, output ONLY the final JSON:"""

    elif task == "funsd":
        planner = f"""You are Form Section Reader 1 (Focus: Header Fields).

Task: Extract form headers and sections.

Form content:
{question}

Instructions:
- Identify header and section fields
- Store findings in latent format

Section 1 extraction:"""

        critic = f"""You are Form Section Reader 2 (Focus: Question Fields).

Task: Extract question labels.

Instructions:
- Identify all question/prompt fields
- Store findings in latent format

Section 2 extraction:"""

        refiner = f"""You are Form Section Reader 3 (Focus: Answer Fields).

Task: Extract answer values and link to questions.

Instructions:
- Identify answer fields
- Match with corresponding questions
- Store findings in latent format

Section 3 extraction:"""

        judger = f"""You are Integration Agent. Consolidate form structure.

Format:
{{"entities": [{{"text": "...", "label": "question|answer|header|other"}}], "relations": [{{"head": "question text", "tail": "answer text"}}]}}

Based on all sections, output ONLY the final JSON:"""

    elif task == "cord":
        planner = f"""You are Receipt Section Reader 1 (Focus: Menu Items).

Task: Extract menu items from receipt.

Receipt content:
{question}

Instructions:
- Identify all menu items and prices
- Store findings in latent format

Items extraction:"""

        critic = f"""You are Receipt Section Reader 2 (Focus: Subtotals).

Task: Extract subtotal and service charges.

Instructions:
- Identify subtotal amounts
- Note service charges
- Store findings in latent format

Subtotals extraction:"""

        refiner = f"""You are Receipt Section Reader 3 (Focus: Tax and Total).

Task: Extract tax and total amount.

Instructions:
- Identify tax amount
- Locate final total
- Store findings in latent format

Tax/Total extraction:"""

        judger = f"""You are Integration Agent. Consolidate receipt data.

Format:
{{"num_items": N, "subtotal_price": "X.XX", "service_price": "", "tax_price": "", "total_price": "X.XX", "etc": ""}}

Based on all sections, output ONLY the final JSON:"""

    elif task == "finer":
        planner = f"""You are Text Section Reader 1 (Focus: Organization Entities).

Task: Extract ORG, LOC entities from first part.

Text:
{question}

Entity types focus: ORG, LOC

Instructions:
- Identify organization and location entities
- Note boundaries
- Store findings in latent format

Section 1 extraction:"""

        critic = f"""You are Text Section Reader 2 (Focus: Financial Entities).

Task: Extract MONEY, PERCENT, DATE entities from middle part.

Entity types focus: MONEY, PERCENT, DATE

Instructions:
- Identify monetary amounts, percentages, dates
- Note boundaries
- Store findings in latent format

Section 2 extraction:"""

        refiner = f"""You are Text Section Reader 3 (Focus: Specialized Entities).

Task: Extract PER, STOCK, METRIC entities from final part.

Entity types focus: PER, STOCK, METRIC, PRODUCT, LAW

Instructions:
- Identify person names, stocks, metrics
- Note boundaries
- Store findings in latent format

Section 3 extraction:"""

        judger = f"""You are Integration Agent. Consolidate all entities.

Format:
{{"entities": [{{"text": "Apple", "type": "ORG", "start": 0, "end": 5}}]}}

Based on all sections, output ONLY the final JSON:"""

    else:
        planner = f"Section 1:\n{question}"
        critic = "Section 2 analysis."
        refiner = "Section 3 analysis."
        judger = "Output final JSON:"
    
    return {
        "planner": planner,
        "critic": critic,
        "refiner": refiner,
        "judger": judger
    }


def generate_agent_reasoning(task: str, gold: str, agent_role: str, mode: str = "sequential") -> str:
    """生成每个agent的推理过程（用于训练）"""
    
    if mode == "hierarchical":
        # Hierarchical模式: 每个agent处理不同partition
        if agent_role == "planner":
            return f"""Partition 1 analysis complete...

Key findings from this section:
- Extracted primary entities/fields
- Noted key information
- Stored in latent representation

[Partition 1 latent state ready]"""

        elif agent_role == "critic":
            return f"""Partition 2 analysis complete...

Key findings from this section:
- Extracted secondary entities/fields
- Cross-referenced with partition 1
- Stored in latent representation

[Partition 2 latent state ready]"""

        elif agent_role == "refiner":
            return f"""Partition 3 analysis complete...

Key findings from this section:
- Extracted remaining entities/fields
- Verified consistency across partitions
- Stored in latent representation

[Partition 3 latent state ready]"""

        elif agent_role == "judger":
            return gold
    
    else:
        # Sequential模式: 顺序协作
        if agent_role == "planner":
            return f"""Scanning document for relevant information...

Key observations:
- Identified main entities and relationships
- Noted evidence locations
- Stored initial findings in latent representation

[Latent state updated with document scan results]"""

        elif agent_role == "critic":
            return f"""Verifying extracted information...

Validation results:
- Cross-checked entity references
- Verified relationship accuracy
- Identified areas needing refinement

[Latent state updated with verification results]"""

        elif agent_role == "refiner":
            return f"""Organizing and refining structure...

Refinement complete:
- Consolidated all verified information
- Resolved conflicts
- Prepared final extraction format

[Latent state finalized for output]"""

        elif agent_role == "judger":
            return gold
    
    return ""


class LatentMASDataset(Dataset):
    """支持4-Agent流程的训练数据集"""
    
    def __init__(self, data_items: List[Dict], processor, task: str, training_mode: str = "latent_mas", prompt_style: str = "sequential"):
        self.data_items = data_items
        self.processor = processor
        self.task = task
        self.training_mode = training_mode
        self.prompt_style = prompt_style  # sequential 或 hierarchical
        self.agents = ["planner", "critic", "refiner", "judger"]
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        question = item.get("question", "")
        gold = item.get("gold", "{}")
        image = item.get("image")
        entity_list = item.get("entity_list", "")
        
        system_msg = "You are Qwen, an expert document extraction assistant. Follow the multi-agent pipeline: scan → verify → organize → output."
        
        if self.training_mode == "latent_mas":
            # 完整4-Agent多轮对话
            messages = self._build_latent_mas_messages(question, gold, entity_list, system_msg, image)
        else:
            # 单轮直接推理 (fallback)
            messages = self._build_direct_messages(question, gold, entity_list, system_msg, image)
        
        return self._process_messages(messages, image)
    
    def _build_latent_mas_messages(self, question: str, gold: str, entity_list: str, 
                                    system_msg: str, image) -> List[Dict]:
        """构建完整的4-Agent对话序列"""
        
        agent_prompts = get_agent_prompts(self.task, question, entity_list, mode=self.prompt_style)
        
        messages = [{"role": "system", "content": system_msg}]
        
        # Planner轮次 (可包含图像)
        if image:
            planner_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": agent_prompts["planner"]}
            ]
        else:
            planner_content = agent_prompts["planner"]
        
        messages.append({"role": "user", "content": planner_content})
        messages.append({"role": "assistant", "content": generate_agent_reasoning(self.task, gold, "planner", mode=self.prompt_style)})
        
        # Critic轮次
        messages.append({"role": "user", "content": agent_prompts["critic"]})
        messages.append({"role": "assistant", "content": generate_agent_reasoning(self.task, gold, "critic", mode=self.prompt_style)})
        
        # Refiner轮次
        messages.append({"role": "user", "content": agent_prompts["refiner"]})
        messages.append({"role": "assistant", "content": generate_agent_reasoning(self.task, gold, "refiner", mode=self.prompt_style)})
        
        # Judger轮次 - 输出最终JSON
        messages.append({"role": "user", "content": agent_prompts["judger"]})
        messages.append({"role": "assistant", "content": gold})  # 最终答案
        
        return messages
    
    def _build_direct_messages(self, question: str, gold: str, entity_list: str,
                                system_msg: str, image) -> List[Dict]:
        """构建单轮直接推理对话"""
        
        # 使用原来的单轮prompt格式
        instruction = self._get_direct_instruction(entity_list)
        
        if image:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{instruction}\n\nDocument text:\n{question}\n\nExtract and output JSON:"}
            ]
        else:
            user_content = f"{instruction}\n\nDocument text:\n{question}\n\nExtract and output JSON:"
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gold}
        ]
    
    def _get_direct_instruction(self, entity_list: str) -> str:
        """获取单轮推理的指令"""
        if self.task == "docred":
            return f"""Task: Document-level relation extraction.
Entities: {entity_list}
Output: {{"relations": [{{"head": "...", "relation": "P-ID", "tail": "...", "evidence": [0]}}]}}"""
        elif self.task == "funsd":
            return """Task: Form field extraction.
Output: {"entities": [...], "relations": [...]}"""
        elif self.task == "cord":
            return """Task: Receipt extraction.
Output: {"num_items": N, "subtotal_price": "...", "total_price": "..."}"""
        elif self.task == "finer":
            return """Task: Financial NER.
Output: {"entities": [{"text": "...", "type": "...", "start": 0, "end": 5}]}"""
        return "Extract information."
    
    def _process_messages(self, messages: List[Dict], image) -> Dict:
        """处理消息并返回模型输入"""
        
        # 应用chat template
        if hasattr(self.processor, 'apply_chat_template'):
            text = self.processor.apply_chat_template(messages, tokenize=False)
        else:
            text = self._manual_format_messages(messages)
        
        # Tokenize
        if image:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=4096,  # 4-agent需要更长上下文
                truncation=True
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding="max_length",
                max_length=4096,
                truncation=True
            )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # 创建labels
        labels = input_ids.clone()
        pad_token_id = getattr(self.processor, 'pad_token_id', None) or \
                       getattr(self.processor.tokenizer, 'pad_token_id', 0)
        labels[labels == pad_token_id] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"].squeeze(0)
        
        return result
    
    def _manual_format_messages(self, messages: List[Dict]) -> str:
        """手动格式化消息（fallback）"""
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
            text_parts.append(f"<|{role}|>\n{content}")
        return "\n".join(text_parts)


# Backward compatibility: keep old class name
DocumentExtractionDataset = LatentMASDataset


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
    parser.add_argument("--training_mode", type=str, default="direct", choices=["latent_mas", "direct"],
                       help="Training mode: 'direct' for task-only (recommended), 'latent_mas' for agent-aware")
    parser.add_argument("--prompt_style", type=str, default="sequential", choices=["sequential", "hierarchical"],
                       help="[Only for latent_mas mode] Prompt style: 'sequential' or 'hierarchical'")
    args = parser.parse_args()
    
    print(f"[Config] Training mode: {args.training_mode}")
    if args.training_mode == "direct":
        print("  → Task-only training: LoRA learns 'Document → JSON' mapping")
        print("  → Can be used with ANY LatentMAS mode at inference (sequential/hierarchical)")
        print("  → Recommended: More flexible, agent roles assigned at inference time")
    else:
        print(f"  → Agent-aware training: LoRA learns full 4-agent flow ({args.prompt_style} style)")
        print(f"  → Must use same prompt style at inference")
        print(f"  → Use case: When you want agents 'baked into' model weights")
    
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
    
    train_dataset = LatentMASDataset(train_data, processor, args.task, training_mode=args.training_mode, prompt_style=args.prompt_style)
    
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
