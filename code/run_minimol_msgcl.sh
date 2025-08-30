#!/bin/bash
#SBATCH -c 1
#SBATCH -t 1:10:00
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o run_logs/minimol_%j.out
#SBATCH -e run_logs/minimol_%j.err

# -------------------------------
# SLURM job for Minimol grid search
# -------------------------------

module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
eval "$(conda shell.bash hook)"
conda activate minimol-env


WORKDIR=".."
SCRIPT="$WORKDIR/code/minimol_triplet_runner.py"  
INPUT_CSV="data/lipophilicity_150_train.csv"
OUTPUT_BASE="Outputs"
#INPUT_CSV= "data/lipophilicity_150_train_val_from_test_minTestSim.csv"
# -------------------------------
# Metadata
# -------------------------------
RUN_ID="lipophilicity_minimol_triplet$(date +%Y%m%d_%H%M%S)"
TASK="lipophilicity"  # choices: lipophilicity | ames


cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"


python -u code/minimol_triplet_runner.py \
  --csv_path $INPUT_CSV \
  --output_dir $OUTPUT_BASE/$RUN_ID \
  --cache_file "/n/data1/hms/dbmi/farhat/anz226/Lyme_AZ/cache/smiles_fp_cache.pt" \
  --lrs "3e-4" \
  --epochs "100" \
  --batch_size 128 \
  --margins "0.25" \
  --triplet_weights "200" \
  --max_frags_per_mol 30\
  --replicates 25 \
  --task "$TASK" \
  --min_score 3 # for lipophilicity. 

