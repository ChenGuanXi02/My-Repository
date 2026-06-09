## !/bin/bash
#
## 是否启用多卡
#MULTI_CARD=true
#
## 随机运行 ID 和结果保存路径
#RANDOM_RUN_ID=$(shuf -i 100000-999999 -n 1)
#RESULTS_FOLDER="results/CamoDiffusion_run_${RANDOM_RUN_ID}"
#
## 采样步数和批量大小
#SAMPLING_STEPS=(10) # 可添加更多步数
#BATCH_SIZE=16
#
## 数据集目标，多个数据集可用空格分隔
##TARGET_DATASETS=("COD10K")
#TARGET_DATASETS=("CASIAv1 Columbia Coverage IMD20 NIST16")
#
## 默认启动命令
#LAUNCH_COMMAND="python"
#if ${MULTI_CARD}; then
#    RANDOM_PORT=$(shuf -i 20000-29999 -n 1)
#    LAUNCH_COMMAND="accelerate launch --num_machine=1 --multi_gpu --num_processes=2 --gpu_ids=2,3 --mixed_precision=no --main_process_port ${RANDOM_PORT}"
#fi
#
## 配置文件和检查点路径
#CONFIG_FILE="config/camoDiffusion_384x384.yaml" # 修改为实际配置路径
#CHECKPOINT="newmodel_finetune/model-best.pt" # 修改为实际模型权重路径
#
## 创建结果文件夹
#mkdir -p ${RESULTS_FOLDER}
#
## 遍历数据集并运行测试
#for DATASET in "${TARGET_DATASETS[@]}"; do
#    echo "Testing dataset: ${DATASET} with ${SAMPLING_STEPS[0]} steps"
#    ${LAUNCH_COMMAND} sample.py \
#        --config ${CONFIG_FILE} \
#        --results_folder ${RESULTS_FOLDER}/${DATASET}_result \
#        --checkpoint ${CHECKPOINT} \
#        --num_sample_steps ${SAMPLING_STEPS[0]} \
#        --target_dataset ${DATASET} \
#        --time_ensemble
#done


MULTI_CARD=true
RANDOM_RUN_ID=$(shuf -i 100000-999999 -n 1)
RESULTS_FOLDER="results/CamoDiffusion_run_${RANDOM_RUN_ID}"


SAMPLING_STEPS=(10)
BATCH_SIZE=8
TARGET_DATASET="LR"
# CASIAv1 Coverage_25 NIST16_160
# CASIAv1 Columbia Coverage IMD20 NIST16
# CNN-Based-800 GAN-Based-800 TM-Based-800 DM-Based-800
# GC CA SH EC LB RN NS LR PM SG

LAUNCH_COMMAND="python"
if ${MULTI_CARD}; then
    RANDOM_PORT=$(shuf -i 20000-29999 -n 1)
    LAUNCH_COMMAND="accelerate launch --num_machine=1 --multi_gpu --num_processes=2 --gpu_ids=2,3 --mixed_precision=no --main_process_port ${RANDOM_PORT}"
#    LAUNCH_COMMAND="accelerate launch --num_machine=1 --gpu_ids=2 --mixed_precision=no --main_process_port ${RANDOM_PORT}"
fi

CONFIG_FILE="config/260301Inpainting32K/Train_352.yaml"
#CONFIG_FILE="config/TPAMI_TrainFinetune/Train_2w_352.yaml"
#CONFIG_FILE="config/TPAMI_TrainFinetune/Fine_Casia_352.yaml"
#CONFIG_FILE="config/TPAMI_TrainFinetune/Fine_Cover_352.yaml"
#CONFIG_FILE="config/TPAMI_TrainFinetune/Fine_Nist_352.yaml"
#CONFIG_FILE="config/Casiav2TrainDefacto6kSample/Train_Casiav2_352.yaml"

# CHECKPOINT="/home/chenzeyu/pycharmprojects/CamoDiffusion/model_352_v3/model-91.pt"
CHECKPOINT="/home/chenzeyu/pycharmprojects/CamoDiffusion/results/In32K-bs26-lr1e-4-epoch160-edge_weight=0.1/special_checkpoints/model-best.pt"

# echo Config file and checkpoint
echo -e "\033[32m [Config file: ${CONFIG_FILE}] \033[0m"
echo -e "\033[32m [Checkpoint: ${CHECKPOINT}] \033[0m"

for i in "${SAMPLING_STEPS[@]}";do
    # Check if the results folder exists, if so, delete it
    if [ -d "${RESULTS_FOLDER}" ]; then
        echo -e "\033[31m [Results folder ${RESULTS_FOLDER} exists, deleting...] \033[0m"
        rm -rf ${RESULTS_FOLDER}
    else
        echo -e "\033[32m [Results folder ${RESULTS_FOLDER} does not exist, creating...] \033[0m"
        mkdir -p ${RESULTS_FOLDER}
    fi
    echo -e "\033[32m Sampling ${i} steps \033[0m"
    ${LAUNCH_COMMAND} sample.py \
      --config=${CONFIG_FILE} \
      --results_folder=${RESULTS_FOLDER} \
      --checkpoint=${CHECKPOINT} \
      --batch_size=${BATCH_SIZE} \
      --num_sample_steps=${i} \
      --target_dataset ${TARGET_DATASET} \
      --time_ensemble \
#      --vis_priors \
#      --vis_n 180 \
#      --vis_dir vis/priors\
#      --vis_overlay \
#      --vis_overlay_alpha 0.7
done