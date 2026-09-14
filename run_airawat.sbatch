#!/bin/bash
#SBATCH --job-name=autored_uday
#SBATCH --partition=airawatp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm_uday_%j.out
#SBATCH --error=slurm_uday_%j.err

echo "=========================================================="
echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# HuggingFace Offline Environment for AIRAWAT compute nodes
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
export HF_HOME="${HF_HOME:-/nlsasfs/home/isea/isea30/.cache/huggingface}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Conda activation
source /nlsasfs/home/isea/isea30/anaconda3/etc/profile.d/conda.sh
conda activate spectre || conda activate redteam

cd "/nlsasfs/home/isea/isea30/MTP/ARed-main_Uday(ours)"

# Step 1: Supervised Fine-Tuning (Attacker)
echo "--- [1/4] Running SFT Training ---"
python -u train_sft.py

# Step 2: Stop-Point Identifier (BERT Detector)
echo "--- [2/4] Running Detector Training ---"
python -u train_detector.py

# Step 3: Sensitive Info Extractor (T5)
echo "--- [3/4] Running Extractor Training ---"
python -u train_extractor.py

# Step 4: NLPO/PPO RL Training & 4-LLM CTF Evaluation
echo "--- [4/4] Running RL Training & Evaluation ---"
python -u run_autored.py

echo "=========================================================="
echo "Job finished at $(date)"
echo "=========================================================="
