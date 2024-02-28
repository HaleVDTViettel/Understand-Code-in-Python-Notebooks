## Overview

I used single list-wise deberta-v3-large trained on all competition data. Input text is cells separated with [SEP] and [CLS] tokens.

## What Worked

- MLM
- preprocessing
    - Reasoning and context: It makes text shorter, so we can handle longer text.
- use deberta rather than codebert
- +lstm head
- bigger cell_cnt and seq_length
    - Reasoning and context: metric function cares more about long text.
- +postprocessing: predict code and markdown both, then sort code by ground truth.
    - Reasoning and context: more reasonable than only predict markdown, and concat with code cells.

## What Didn’t Work

- translation: facebook/mbart-large-50-many-to-one-mmt
    - Reasoning and context: most notebooks are in english. Or I did something wrong.
- +code rank to dense layers
    - Reasoning and context: I should try to concat text with rank information. Model will learn better.
- extract embedding, then train a transformer model
    - Reasoning and context: I should pre-train the extracting model first.
- xlm-robert-large or mdeberta-v3-base
    - Reasoning and context: much worse than deberta-v3-large. so can not ensemble directly.

## Additional Context

- loss is MAE.
- MLM on training data for 15 epochs, max_length is 1024. It takes 3 days.
- training for 10 epochs, max_length is 2048. It takes 14 days. I think set max_length to 5120 will greatly improve model performance but I have not enough rtx3090~
- inference: max_length is 5120. It takes 6 hours.
- I used 1 * 3090(24GB) at the early stage.
- Future work:
    - concat text with rank information: flag + rank_1 + cell_1 + flag+ rank_2 + cell_2 + … + flag+ rank_n + cell_n
    - multi languages
    - extra training data
    - Adversarial Weight Perturbations (AWP), but too time-consuming

## ARCHIVE CONTENTS
input : dataset

src: code to rebuild models from scratch 

## HARDWARE: (The following specs were used to create the original solution)
Inference & Training:
OS: Ubuntu 22.04.3 LTS x86_64
GPU: 1 * RTX3090
CPU: 1 * 13th Gen Intel i5-13600K (20) @ 5.100GHz
RAM: 32GB DDR5

## Configuration files & SETTINGS.json
see src/parameter.py

## MODEL TRAIN&PREDICT
1.Training:

cd src

python train_mlm.py --model_name deberta-v3-large --base_epoch 15 --batch_size 7 --learning_rate 5e-6 --max_length 1024

python train.py --model deberta-v3-large --base_epoch 10 --batch_size 5 --lr 5e-6 --seq_length 2048 --max_grad_norm 1.0 --folds 0  

2.Inference: