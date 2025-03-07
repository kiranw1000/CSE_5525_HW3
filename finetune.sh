#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS2912
#SBATCH --ntasks-per-node=1 
#SBATCH--gres=gpu:v100:1

#   A Basic Python Serial Job for the OSC Pitzer cluster
#   https://www.osc.edu/resources/available_software/software_list/python

#
# The following lines set up the Python environment
#
module load python/3.9-2022.05
#
# Move to the directory where the job was submitted from
# You could also 'cd' directly to your working directory
#
# Run Python
#
python train_t5.py --experiment_name FTBase --finetune --learning_rate 1e-4 --batch_size 16 --patience_epochs 10 --max_n_epochs 50
python train_t5.py --experiment_name FTB8 --finetune --learning_rate 1e-4 --batch_size 8 --patience_epochs 10 --max_n_epochs 50
python train_t5.py --experiment_name FTLinear --finetune --learning_rate 1e-4 --batch_size 16 --patience_epochs 10 --max_n_epochs 50 --scheduler_type linear
python train_t5.py --experiment_name FTLinearB8 --finetune --learning_rate 1e-4 --batch_size 8 --patience_epochs 10 --max_n_epochs 50 --scheduler_type linear
python train_t5.py --experiment_name FTSmLR --finetune --learning_rate 1e-5 --batch_size 16 --patience_epochs 10 --max_n_epochs 50
python train_t5.py --experiment_name FTCos --finetune --learning_rate 1e-4 --batch_size 16 --patience_epochs 10 --max_n_epochs 50 --scheduler_type cosine
python train_t5.py --experiment_name FTCosB8 --finetune --learning_rate 1e-4 --batch_size 8 --patience_epochs 10 --max_n_epochs 50 --scheduler_type cosine