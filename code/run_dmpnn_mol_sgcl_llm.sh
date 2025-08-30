#!/bin/bash
#SBATCH -c 1
#SBATCH -t 6:00:00
#SBATCH -p short
#SBATCH --mem=50G
#SBATCH -o run_logs/hostname_%j.out
#SBATCH -e run_logs/hostname_%j.err


module load conda/miniforge3/24.11.3-0
module load gcc/14.2.0
eval "$(conda shell.bash hook)"
conda activate chempropv2
# Paths
WORKDIR=".."
SCRIPT="$WORKDIR/code/dmpnn_mol_sgcl.py"
OUTPUT_BASE="Outputs/"
INPUT_CSV="data/lipophilicity_150_train.csv"
#INPUT_CSV="/n/data1/hms/dbmi/farhat/anz226/Unlearning_Workshop/data/final_data_splits/lipophilicity_150_train_val_from_test_minTestSim.csv"

# Metadata
RUN_ID="lipophilicity_sgcl_dmpnn_llm_1e_3_300epoch$(date +%Y%m%d_%H%M%S)"

cd "$WORKDIR"
mkdir -p "$OUTPUT_BASE/$RUN_ID"

python -u "code/dmpnn_mol_sgcl.py" \
  --run_id "$RUN_ID" \
  --input_csv_path "$INPUT_CSV" \
  --output_dir "$OUTPUT_BASE"\
  --is_regression \
  --task "lipophilicity"\
  --max_molecules_for_plausibility 100 \
  --total_epochs "300" \
  --reinterpretations "15" \
  --learning_rates "1e-3" \
  --triplet_weights "40" \
  --margins "0.25"\
  --n_replicates "6"\
  --use_llm_plausibility \
  --min_value_regression 3\
  --task_description "Lipophilic Molecule" \
  --dataset_description "Lipophilicity is a measure of how easily a chemical dissolves in a non-polar solvent." \


