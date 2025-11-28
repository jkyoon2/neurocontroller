#!/bin/bash
# VAE 데이터 수집 배치 스크립트
# 여러 레이아웃, 시드, 체크포인트에 대해 데이터 수집

set -e  # 에러 발생 시 중단

# 기본 설정
# RESULTS_DIR: 학습된 모델 체크포인트가 저장된 디렉토리
# 체크포인트 파일 경로: ${RESULTS_DIR}/${LAYOUT}/${ALGORITHM}/${EXPERIMENT}/seed${SEED}/models/actor_agent{0,1}_periodic_${STEP}.pt
PROJECT_ROOT="/home/juliecandoit98/ZSC-Eval"
RESULTS_DIR="${PROJECT_ROOT}/results/Overcooked"  # 체크포인트 파일 위치
VAE_DATA_DIR="${PROJECT_ROOT}/vae_data"  # 수집된 VAE 데이터 저장 위치
ALGORITHM="rmappo"
EXPERIMENT="sp"
NUM_AGENTS=2
EPISODES_PER_CHECKPOINT=5
K_TIMESTEPS=4
ENCODING="OAI_lossless"  # 또는 "OAI_raw_image"
DEVICE="cuda"

# 로그 디렉토리 생성
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../log"
mkdir -p "${LOG_DIR}"

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/collect_vae_data_${TIMESTAMP}.log"

echo "=========================================="
echo "VAE Data Collection Batch Script"
echo "Started at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# 레이아웃별 설정
declare -A LAYOUT_STEPS
LAYOUT_STEPS["random0_medium"]="2020000 2520000 3020000 3520000 4020000 4520000 5020000"
LAYOUT_STEPS["random1"]="2020000 2520000 3020000 3520000 4020000 4520000 5020000"
LAYOUT_STEPS["random3"]="5520000 6020000 6520000 7020000 7520000 8020000 8520000"
LAYOUT_STEPS["small_corridor"]="7020000 7520000 8020000 8520000 9020000 9520000 10000000"

# 시드 리스트
SEEDS=(5 6 7 8 9)

# 각 레이아웃에 대해 처리
for LAYOUT in random0_medium random1 random3 small_corridor; do
    echo "" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    echo "Processing Layout: ${LAYOUT}" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    
    STEPS=(${LAYOUT_STEPS[$LAYOUT]})
    
    # 각 시드에 대해 처리
    for SEED in "${SEEDS[@]}"; do
        echo "" | tee -a "${LOG_FILE}"
        echo "  Processing Seed: ${SEED}" | tee -a "${LOG_FILE}"
        
        # 체크포인트 디렉토리 경로
        CHECKPOINT_BASE_DIR="${RESULTS_DIR}/${LAYOUT}/${ALGORITHM}/${EXPERIMENT}/seed${SEED}/models"
        
        # 체크포인트 디렉토리 존재 확인
        if [ ! -d "${CHECKPOINT_BASE_DIR}" ]; then
            echo "    ⚠ Warning: Checkpoint directory not found: ${CHECKPOINT_BASE_DIR}" | tee -a "${LOG_FILE}"
            continue
        fi
        
        # 각 체크포인트 스텝에 대해 처리
        for STEP in "${STEPS[@]}"; do
            echo "    Processing Step: ${STEP}" | tee -a "${LOG_FILE}"
            
            # Agent0, Agent1 체크포인트 파일 경로
            AGENT0_CHECKPOINT="${CHECKPOINT_BASE_DIR}/actor_agent0_periodic_${STEP}.pt"
            AGENT1_CHECKPOINT="${CHECKPOINT_BASE_DIR}/actor_agent1_periodic_${STEP}.pt"
            
            # 체크포인트 파일 존재 확인
            if [ ! -f "${AGENT0_CHECKPOINT}" ] || [ ! -f "${AGENT1_CHECKPOINT}" ]; then
                echo "      ⚠ Warning: Checkpoint files not found for step ${STEP}" | tee -a "${LOG_FILE}"
                echo "        Agent0: ${AGENT0_CHECKPOINT}" | tee -a "${LOG_FILE}"
                echo "        Agent1: ${AGENT1_CHECKPOINT}" | tee -a "${LOG_FILE}"
                continue
            fi
            
            # 출력 디렉토리 생성
            OUTPUT_DIR="${VAE_DATA_DIR}/${LAYOUT}/seed${SEED}/step${STEP}"
            mkdir -p "${OUTPUT_DIR}"
            
            # 저장 경로
            SAVE_PATH="${OUTPUT_DIR}/buffer_${LAYOUT}_seed${SEED}_step${STEP}.pkl"
            
            # 데이터 수집 명령 실행
            echo "      Collecting data..." | tee -a "${LOG_FILE}"
            echo "        Agent0: ${AGENT0_CHECKPOINT}" | tee -a "${LOG_FILE}"
            echo "        Agent1: ${AGENT1_CHECKPOINT}" | tee -a "${LOG_FILE}"
            echo "        Output: ${SAVE_PATH}" | tee -a "${LOG_FILE}"
            
            # collect_vae_data.py 실행
            # agent0, agent1 체크포인트 파일을 직접 지정
            python -m zsceval.utils.vae.collect_vae_data \
                --layout "${LAYOUT}" \
                --num-agents "${NUM_AGENTS}" \
                --encoding "${ENCODING}" \
                --episodes-random 0 \
                --episodes-per-checkpoint "${EPISODES_PER_CHECKPOINT}" \
                --k-timesteps "${K_TIMESTEPS}" \
                --checkpoint-files "${AGENT0_CHECKPOINT}" "${AGENT1_CHECKPOINT}" \
                --algorithm "${ALGORITHM}" \
                --save-path "${SAVE_PATH}" \
                --seed "${SEED}" \
                --device "${DEVICE}" \
                --obs-width 13 \
                --obs-height 5 \
                2>&1 | tee -a "${LOG_FILE}"
            
            # 실행 결과 확인
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "      ✓ Successfully collected data for ${LAYOUT}/seed${SEED}/step${STEP}" | tee -a "${LOG_FILE}"
            else
                echo "      ✗ Failed to collect data for ${LAYOUT}/seed${SEED}/step${STEP}" | tee -a "${LOG_FILE}"
            fi
            
            # 메모리 정리 (선택사항)
            if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
                python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
            fi
        done
    done
done

echo "" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"
echo "VAE Data Collection Completed"
echo "Finished at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

