#!/bin/bash
# scripts/train_sbatch.sh
# Example SLURM batch script for NYU CIMS (adjust directives to local cluster policy)

#SBATCH --job-name=hinglish-ner
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Load modules if your cluster uses module system (example)
# module load anaconda/2022.05
# module load cuda/12.1

# Activate conda
source ~/.bashrc || true
if command -v conda >/dev/null 2>&1; then
  source $(conda info --base)/etc/profile.d/conda.sh || true
  conda activate hinglish-ner || true
fi

# cd to repo
cd ~/work/NLP-F25-Team-1 || exit 1

# make sure logs dir exists
mkdir -p logs

# Training command: tune batch_size and grad accumulation to your GPU memory
python -m src.train_flat_ner \
  --model_name xlm-roberta-base \
  --output_dir outputs/xlmr-flat-gpu \
  --batch_size 8 \
  --num_train_epochs 5 \
  --gradient_accumulation_steps 4 \
  --fp16 true \
  --report_to none

# End of script