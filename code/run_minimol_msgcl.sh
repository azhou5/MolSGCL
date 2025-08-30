
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
  --cache_file "<INSERT MINIMIOL FINGERPRINT CACHE FILE PATH HERE>" \
  --lrs "3e-4" \
  --epochs "100" \
  --batch_size 128 \
  --margins "0.25" \
  --triplet_weights "400" \
  --max_frags_per_mol 30\
  --replicates 25 \
  --task "$TASK" \
  --min_score 3 # for lipophilicity. 

