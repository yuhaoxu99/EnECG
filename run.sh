#!/bin/bash
#SBATCH --job-name=slurmtest
#SBATCH --mem=200GB
#SBATCH --output=./output/EnECG/flag.log
#SBATCH --gres=gpu:1

cd /local/scratch/yxu81/fairseq-signals/
source venv/bin/activate
python main.py \
        --label flag \
        --task_name classification \
        --num_class 2 \

