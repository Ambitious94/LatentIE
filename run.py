import argparse
import json
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
    load_docred,
    load_cord,
    load_funsd,
    load_finer
)
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from models import ModelWrapper
from utils import auto_device, set_seed
from evaluate_extraction import evaluate_extraction_task, print_evaluation_results
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct

# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas"], required=True,
                        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name to use (e.g. 'Qwen/Qwen3-14B', 'Qwen/Qwen2-VL-7B-Instruct').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa', 'docred', 'cord', 'funsd', 'finer'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--doc_path", type=str, default=None, help="Path to document file (for docred/cord/funsd/finer tasks)")
    parser.add_argument("--extraction_mode", type=str, choices=["chunks", "partitioned"], default="chunks", 
                        help="Document processing mode: 'chunks' for sequential, 'partitioned' for hierarchical")
    parser.add_argument("--chunk_size", type=int, default=3000, help="Characters per chunk for extraction tasks")
    parser.add_argument("--chunk_overlap", type=int, default=300, help="Overlap between chunks")
    parser.add_argument("--num_partitions", type=int, default=3, help="Number of partitions for hierarchical extraction")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON file (e.g., results.json)")
    
    # Multimodal arguments
    parser.add_argument("--use_vision_model", action="store_true", help="Use vision-language model (Qwen-VL) for multimodal tasks")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file for multimodal extraction (CORD/FUNSD)")
    parser.add_argument("--lora_weights", type=str, default=None, help="Path to LoRA weights directory for fine-tuned model")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images for batch processing")
    parser.add_argument("--annotations_dir", type=str, default=None, help="Directory containing annotation files (FUNSD segm_file JSONs)")

    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    # Force batch_size=1 for vision models to avoid image token mismatch
    if args.use_vision_model:
        print("[INFO] Vision model detected, forcing batch_size=1")
        args.generate_bs = 1
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    elif args.task == "docred":
        if not args.doc_path:
            raise ValueError("--doc_path is required for docred task")
        mode = "partitioned" if args.prompt == "hierarchical" and args.extraction_mode == "partitioned" else "chunks"
        dataset_iter = load_docred(
            doc_path=args.doc_path,
            split=args.split,
            mode=mode,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            num_partitions=args.num_partitions
        )
    elif args.task == "cord":
        if not args.doc_path:
            raise ValueError("--doc_path is required for cord task")
        mode = "partitioned" if args.prompt == "hierarchical" and args.extraction_mode == "partitioned" else "chunks"
        dataset_iter = load_cord(
            doc_path=args.doc_path,
            split=args.split,
            mode=mode,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            num_partitions=args.num_partitions,
            image_path=args.image_path  # Pass image path for multimodal
        )
    elif args.task == "funsd":
        if not args.doc_path:
            raise ValueError("--doc_path is required for funsd task")
        mode = "partitioned" if args.prompt == "hierarchical" and args.extraction_mode == "partitioned" else "chunks"
        dataset_iter = load_funsd(
            doc_path=args.doc_path,
            split=args.split,
            mode=mode,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            num_partitions=args.num_partitions,
            image_path=args.image_path,  # Pass image path for multimodal
            annotations_dir=args.annotations_dir,  # Directory for annotation JSONs
            images_dir=args.image_dir  # Directory for images
        )
    elif args.task == "finer":
        if not args.doc_path:
            raise ValueError("--doc_path is required for finer task")
        mode = "partitioned" if args.prompt == "hierarchical" and args.extraction_mode == "partitioned" else "chunks"
        dataset_iter = load_finer(
            doc_path=args.doc_path,
            split=args.split,
            mode=mode,
            chunk_size=args.chunk_size,
            overlap=args.chunk_overlap,
            num_partitions=args.num_partitions
        )
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    # 根据任务类型选择评估方式
    extraction_tasks = ["docred", "cord", "funsd", "finer"]
    
    if args.task in extraction_tasks:
        # 文档抽取任务：使用专门的评估指标
        print("\n" + "="*70)
        print(f"📊 Evaluating {args.task.upper()} Extraction Results...")
        print("="*70)
        
        # 收集金标准
        golds = [p.get("gold", "") for p in preds]
        
        # 计算评估指标
        metrics = evaluate_extraction_task(args.task, preds, golds)
        
        # 打印详细结果
        print_evaluation_results(args.task, metrics)
        
        # 保存结果
        result_summary = {
            "method": args.method,
            "model": args.model_name,
            "task": args.task,
            "prompt": args.prompt,
            "max_samples": len(preds),
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / len(preds), 4) if len(preds) > 0 else 0,
            **metrics  # 包含所有评估指标
        }
    else:
        # 传统QA任务：使用准确率
        acc, correct = evaluate(preds)
        
        result_summary = {
            "method": args.method,
            "model": args.model_name,
            "split": args.split,
            "seed": args.seed,
            "max_samples": args.max_samples,
            "accuracy": acc,
            "correct": correct,
            "total_time_sec": round(total_time, 4),
            "time_per_sample_sec": round(total_time / args.max_samples, 4),
        }
    
    # Print results to stdout
    print(
        json.dumps(
            result_summary,
            ensure_ascii=False,
            indent=2
        )
    )
    
    # Save results to file if output_path is specified
    if args.output_path:
        output_data = {
            "summary": result_summary,
            "predictions": preds
        }
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to {args.output_path}")
    else:
        # Auto-generate output path
        auto_output_path = f"{args.task}_{args.method}_results.json"
        output_data = {
            "summary": result_summary,
            "predictions": preds
        }
        with open(auto_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Results saved to {auto_output_path}")


if __name__ == "__main__":
    main()
