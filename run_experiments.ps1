# DocRED实验批处理脚本

New-Item -ItemType Directory -Force -Path results

# 实验 1: DocRED (LatentMAS Sequential)
python run.py --method latent_mas --model_name Qwen/Qwen2.5-1.5B-Instruct --task docred --doc_path "e:/Edge Download/dev.json" --prompt sequential --latent_steps 256 --temperature 0.3 --max_new_tokens 8192 --max_samples 50 --output_path results/docred_latent_seq.json

# 实验 2: DocRED (LatentMAS Hierarchical)
python run.py --method latent_mas --model_name Qwen/Qwen2.5-1.5B-Instruct --task docred --doc_path "e:/Edge Download/dev.json" --prompt hierarchical --extraction_mode partitioned --num_partitions 3 --latent_steps 256 --temperature 0.3 --max_new_tokens 8192 --max_samples 50 --output_path results/docred_latent_hier.json

# 实验 3: DocRED (TextMAS)
python run.py --method text_mas --model_name Qwen/Qwen2.5-1.5B-Instruct --task docred --doc_path "e:/Edge Download/dev.json" --prompt sequential --temperature 0.3 --max_new_tokens 8192 --max_samples 50 --output_path results/docred_textmas.json

# 实验 4: DocRED (Baseline)
python run.py --method baseline --model_name Qwen/Qwen2.5-1.5B-Instruct --task docred --doc_path "e:/Edge Download/dev.json" --temperature 0.3 --max_new_tokens 8192 --max_samples 50 --output_path results/docred_baseline.json

