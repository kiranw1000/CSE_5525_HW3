#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --account=PAS2912
#SBATCH --ntasks-per-node=1 
#SBATCH--gres=gpu:v100:1

#   A Basic Python Serial Job for the OSC Pitzer cluster
#   https://www.osc.edu/resources/available_soSRCware/soSRCware_list/python

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
python train_t5.py --experiment_name SRCBase --learning_rate 1e-4 --batch_size 16 --patience_epochs 10 --max_n_epochs 50
python train_t5.py --experiment_name SRCB8 --learning_rate 1e-4 --batch_size 8 --patience_epochs 10 --max_n_epochs 50
python train_t5.py --experiment_name SRCLinear --learning_rate 1e-4 --batch_size 16 --patience_epochs 10 --max_n_epochs 50 --scheduler_type linear
python train_t5.py --experiment_name SRCLinearB8 --learning_rate 1e-4 --batch_size 8 --patience_epochs 10 --max_n_epochs 50 --scheduler_type linear
python train_t5.py --experiment_name SRCSmLR --learning_rate 1e-5 --batch_size 16 --patience_epochs 10 --max_n_epochs 50
python train_t5.py --experiment_name SRCCos --learning_rate 1e-4 --batch_size 16 --patience_epochs 10 --max_n_epochs 50 --scheduler_type cosine
python train_t5.py --experiment_name SRCCosB8 --learning_rate 1e-4 --batch_size 8 --patience_epochs 10 --max_n_epochs 50 --scheduler_type cosine