#!/bin/bash
# VAE 학습 스크립트
# 수집된 VAE 데이터를 사용하여 Temporal VAE 모델 학습

set -e  # 에러 발생 시 중단

# 기본 설정
PROJECT_ROOT="/home/juliecandoit98/ZSC-Eval"
VAE_DATA_DIR="${PROJECT_ROOT}/vae_data"  # VAE 데이터 디렉토리
VAE_MODELS_DIR="${PROJECT_ROOT}/vae_models"  # 학습된 모델 저장 디렉토리
VAE_CHECKPOINTS_DIR="${PROJECT_ROOT}/vae_checkpoints"  # 체크포인트 저장 디렉토리

# 학습 설정
ENCODING="OAI_lossless"  # 인코딩 스킴
K_TIMESTEPS=4  # 시계열 타임스텝 수 (K=4이면 k+1=5개의 타임스텝)
HIDDEN_DIM=128  # Latent space 차원
EPOCHS=100  # 학습 에폭 수
BATCH_SIZE=32  # 배치 크기
LEARNING_RATE=1e-3  # 학습률
BETA=1.0  # KL divergence 가중치
MODE="reconstruction"  # VAE 모드 (reconstruction 또는 predictive)
DEVICE="cuda"  # 디바이스

# Wandb 설정
USE_WANDB=true
WANDB_PROJECT="vae-overcooked"
WANDB_RUN_NAME=""  # 자동 생성 (None이면 자동 생성)

# 로그 디렉토리 생성
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../log"
mkdir -p "${LOG_DIR}"

# 모델 저장 디렉토리 생성
mkdir -p "${VAE_MODELS_DIR}"
mkdir -p "${VAE_CHECKPOINTS_DIR}"

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_vae_${TIMESTAMP}.log"

# Wandb run 이름 자동 생성 (지정되지 않은 경우)
if [ -z "${WANDB_RUN_NAME}" ]; then
    WANDB_RUN_NAME="vae_${ENCODING}_${MODE}_h${HIDDEN_DIM}_k${K_TIMESTEPS}_${TIMESTAMP}"
fi

# 모델 저장 경로
SAVE_PATH="${VAE_MODELS_DIR}/vae_${ENCODING}_${MODE}_h${HIDDEN_DIM}_k${K_TIMESTEPS}_${TIMESTAMP}.pt"
ENCODER_PATH="${VAE_MODELS_DIR}/encoder_vae_${ENCODING}_${MODE}_h${HIDDEN_DIM}_k${K_TIMESTEPS}_${TIMESTAMP}.pt"

echo "=========================================="
echo "VAE Training Script"
echo "Started at: $(date)"
echo "Log file: ${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  VAE Data Dir: ${VAE_DATA_DIR}" | tee -a "${LOG_FILE}"
echo "  Encoding: ${ENCODING}" | tee -a "${LOG_FILE}"
echo "  K Timesteps: ${K_TIMESTEPS}" | tee -a "${LOG_FILE}"
echo "  Hidden Dim: ${HIDDEN_DIM}" | tee -a "${LOG_FILE}"
echo "  Mode: ${MODE}" | tee -a "${LOG_FILE}"
echo "  Epochs: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "  Batch Size: ${BATCH_SIZE}" | tee -a "${LOG_FILE}"
echo "  Learning Rate: ${LEARNING_RATE}" | tee -a "${LOG_FILE}"
echo "  Beta: ${BETA}" | tee -a "${LOG_FILE}"
echo "  Device: ${DEVICE}" | tee -a "${LOG_FILE}"
if [ "${USE_WANDB}" = true ]; then
    echo "  Wandb Project: ${WANDB_PROJECT}" | tee -a "${LOG_FILE}"
    echo "  Wandb Run Name: ${WANDB_RUN_NAME}" | tee -a "${LOG_FILE}"
fi
echo "  Model Save Path: ${SAVE_PATH}" | tee -a "${LOG_FILE}"
echo "  Encoder Save Path: ${ENCODER_PATH}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

# VAE 학습 실행
echo "" | tee -a "${LOG_FILE}"
echo "Starting VAE training..." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# 기본 명령어 구성
CMD="python -m zsceval.utils.vae.train_vae \
    --vae-data-dir ${VAE_DATA_DIR} \
    --layout all \
    --encoding ${ENCODING} \
    --k-timesteps ${K_TIMESTEPS} \
    --hidden-dim ${HIDDEN_DIM} \
    --mode ${MODE} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --beta ${BETA} \
    --save-path ${SAVE_PATH} \
    --save-encoder-path ${ENCODER_PATH} \
    --checkpoint-save-dir ${VAE_CHECKPOINTS_DIR} \
    --device ${DEVICE}"

# Wandb 옵션 추가
if [ "${USE_WANDB}" = true ]; then
    CMD="${CMD} --use-wandb --wandb-project ${WANDB_PROJECT} --wandb-run-name ${WANDB_RUN_NAME}"
fi

# 명령어 실행 및 로깅
echo "Command: ${CMD}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

${CMD} 2>&1 | tee -a "${LOG_FILE}"

# 실행 결과 확인
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    echo "✓ VAE Training Completed Successfully" | tee -a "${LOG_FILE}"
    echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
    echo "  Model saved: ${SAVE_PATH}" | tee -a "${LOG_FILE}"
    echo "  Encoder saved: ${ENCODER_PATH}" | tee -a "${LOG_FILE}"
    echo "  Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    exit 0
else
    echo "" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    echo "✗ VAE Training Failed" | tee -a "${LOG_FILE}"
    echo "Finished at: $(date)" | tee -a "${LOG_FILE}"
    echo "  Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
    echo "==========================================" | tee -a "${LOG_FILE}"
    exit 1
fi

