#!/bin/bash
env="Overcooked"

# Arguments
source_layout=$1     # Layout from which to load models (e.g., random0_medium)
target_layout=$2     # Layout for generalization training (e.g., random3)
cuda_device=${3:-0}  # Third argument for CUDA device, default to 0
checkpoint_step=${4:-10000000}  # Fourth argument for checkpoint step, default to 10000000

# Check if required arguments are provided
if [ -z "$source_layout" ] || [ -z "$target_layout" ]; then
    echo "Error: source_layout and target_layout are required"
    echo "Usage: $0 <source_layout> <target_layout> [cuda_device] [checkpoint_step]"
    echo "Example: $0 random0_medium random3 0 10000000"
    echo ""
    echo "This will:"
    echo "  - Load models from: results/Overcooked/<source_layout>/rmappo/sp/seed{X}/models/actor_agent{Y}_periodic_<checkpoint_step>.pt"
    echo "  - Train on: <target_layout>"
    echo "  - Save to: results/Overcooked/<target_layout>/rmappo/sp_generalize/seed{X}/"
    exit 1
fi

# Always use NEW version for all layouts
version="new"

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"
if [[ "${target_layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps="1e7"

num_agents=2
algo="rmappo"
exp="sp_generalize"  # Changed from "sp" to "sp_generalize"
seed_begin=5
seed_max=10
ulimit -n 65536

# Heterogeneous Shaped Reward weights (30 components for new version)
# 세 에이전트 모두 같은 weights 사용
# w0="0,0,0,-1,0.1,0,0,0,0.1,0,-1,0.5,1.0,0.2,0.1,0.2,-2,-1,0.2,0,0.5,50,0,20,-2,-0.01,-0.01,-0.1,-0.1,50"
# w1="0.1,0,0.2,0,0,0.1,0,0,0,0.2,0,0,0,0,0,0,0,-1,0,0,0,50,0,0,0,-0.01,-0.01,-0.1,-0.1,50"
# w2="0,0,0,-1,0.1,0,0,0,0.1,0,-1,0.5,1.0,0.2,0.1,0.2,-2,-1,0.2,0,0.5,50,0,20,-2,-0.01,-0.01,-0.1,-0.1,50"
w0="0,0,0,0,0,0.1,0,0,0,0.1,0,3,0,10,-2,3,2,2,-2,-2,5,0,0,20,-5,0,0,20,-5,-0.01,-0.01,-0.1,-0.1,30"
w1="0,0,0,0,0,0.1,0,0,0,0.1,0,3,0,10,-2,3,2,2,-2,-2,5,0,0,20,-5,0,0,20,-5,-0.01,-0.01,-0.1,-0.1,30"

echo "========================================"
echo "Generalization Training from Checkpoint"
echo "========================================"
echo "Source layout: ${source_layout}"
echo "Target layout: ${target_layout}"
echo "Algorithm: ${algo}, Experiment: ${exp}"
echo "Seeds: ${seed_begin} to ${seed_max}"
echo "Using CUDA device: ${cuda_device}"
echo "Checkpoint step: ${checkpoint_step}"
echo "Using SEPARATED runner (independent policies per agent)"
echo ""
echo "For each seed X, will load:"
echo "  - results/${env}/${source_layout}/${algo}/sp/seedX/models/actor_agent0_periodic_${checkpoint_step}.pt"
echo "  - results/${env}/${source_layout}/${algo}/sp/seedX/models/actor_agent1_periodic_${checkpoint_step}.pt"
echo ""
echo "Results will be saved to:"
echo "  - results/${env}/${target_layout}/${algo}/${exp}/seedX/"
echo "========================================"

for seed in $(seq ${seed_begin} ${seed_max});
do
    # Construct model paths for current seed
    base_model_dir="../../../results/${env}/${source_layout}/${algo}/sp/seed${seed}/models"
    model_agent0="${base_model_dir}/actor_agent0_periodic_${checkpoint_step}.pt"
    model_agent1="${base_model_dir}/actor_agent1_periodic_${checkpoint_step}.pt"
    
    echo ""
    echo "========== Training seed ${seed} =========="
    echo "Loading models:"
    echo "  Agent 0: ${model_agent0}"
    echo "  Agent 1: ${model_agent1}"
    
    # Check if model files exist
    if [ ! -f "${model_agent0}" ]; then
        echo "ERROR: Model file not found: ${model_agent0}"
        echo "Skipping seed ${seed}"
        continue
    fi
    if [ ! -f "${model_agent1}" ]; then
        echo "ERROR: Model file not found: ${model_agent1}"
        echo "Skipping seed ${seed}"
        continue
    fi
    
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${target_layout} --num_agents ${num_agents} \
    --agent_policy_names ppo ppo \
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 50 --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 ${w0} --w1 ${w1} --share_policy --random_index \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
    --use_render --save_gifs --n_render_rollout_threads 1 --render_episodes 1 \
    --cuda_device ${cuda_device} \
    --wandb_name "kyungyoon" \
    --model_dir_agent0 ${model_agent0} \
    --model_dir_agent1 ${model_agent1}
    
    echo "Finished training seed ${seed}"
    echo ""
done

echo "========================================"
echo "All seeds completed!"
echo "======================================"

