cd src

python train_mlm.py \
    --model_name microsoft/deberta-v3-large \
    --base_epoch 15 \
    --batch_size 7 \
    --learning_rate 5e-6 \
    --max_length 1024

python train.py \
    --model microsoft/deberta-v3-large \
    --base_epoch 10 \
    --batch_size 5 \
    --lr 5e-6 \
    --seq_length 2048 \
    --max_grad_norm 1.0 \
    --folds 0  