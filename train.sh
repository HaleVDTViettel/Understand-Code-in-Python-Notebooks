 python src/train.py \
 --model_name_or_path microsoft/codebert-base \
 --md_max_len 128 \
 --total_max_len 512 \
 --batch_size 42 \
 --accumulation_steps 128 \
 --epochs 7 \
 --n_workers 8