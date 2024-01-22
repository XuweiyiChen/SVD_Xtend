#!/bin/bash

#SBATCH --account=fouhey2
#SBATCH --partition=spgpu
#SBATCH --time=48:00:00  # Adjusted to 24 hours, change as needed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8  # Increased CPU count for parallel processing
#SBATCH --mem=64G  # Adjust as per your job's requirement
#SBATCH --gres=gpu:2  # Adjust based on GPU needs
#SBATCH --job-name="svd1"
#SBATCH --output=/scratch/jjparkcv_root/jjparkcv1/xuweic/SVD_Xtend/train_1_syn.log
#SBATCH --mail-type=BEGIN,END,FAIL  # Removed BEGIN and REQUEUE notifications

accelerate launch train_svd.py \
    --pretrained_model_name_or_path="/nfs/turbo/coe-jjparkcv/xuweic/stable_video_diffusion_robotics_finetune/model/stable-video-diffusion-img2vid" \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=50000 \
    --width=512 \
    --height=320 \
    --checkpointing_steps=5000 \
    --checkpoints_total_limit=10 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --seed=542 \
    --num_frames=9 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --dataset_json_path="/scratch/jjparkcv_root/jjparkcv1/xuweic/data/svd_robomimic" \
    --num_validation_images=10