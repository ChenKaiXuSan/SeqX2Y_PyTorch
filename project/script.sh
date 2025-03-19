#!/bin/bash

# dataset path
BASE_PATH="/mnt/dataset/ouyang/dataset/468"

# script path
TRAIN_SCRIPT="/home/ec2-user/SeqX2Y_PyTorch/project/main.py"

# The original and temporary paths of the configuration file
CONFIG_TEMPLATE="/home/ec2-user/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
TEMP_CONFIG="/home/ec2-user/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# 6 patients
NUM_PATIENTS=6

for i in $(seq 1 $NUM_PATIENTS)
do
    echo "开始第 $i 折交叉验证"

    # 复制模板配置文件到临时配置文件
    cp $CONFIG_TEMPLATE $TEMP_CONFIG

    # 生成4DCT和2D和1D数据路径
    TRAIN_PATHS_4D=()
    TRAIN_PATHS_2D=()
    VAL_PATH_4D="$BASE_PATH/POPI_seq3_468/4DCT-Dicom$i"
    VAL_PATH_2D="$BASE_PATH/POPI_seq3_2D_468/2DCT-$i"
    # PATH_1D="/home/ec2-user/SeqX2Y_PyTorch/dataset/Diagram_Coordinates/1D_rpm.csv"

    for j in $(seq 1 $NUM_PATIENTS)
    do
        if [ $j -ne $i ]
        then
            TRAIN_PATHS_4D+=("$BASE_PATH/POPI_seq3_468/4DCT-Dicom$j")
            TRAIN_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_468/2DCT-$j")
        fi
    done

    # 把训练路径数组转换为以逗号分隔的字符串
    TRAIN_PATHS_4D_STR=$(IFS=,; echo "${TRAIN_PATHS_4D[*]}")
    TRAIN_PATHS_2D_STR=$(IFS=,; echo "${TRAIN_PATHS_2D[*]}")

    # 更新临时配置文件
    sed -i "s|data_path: \".*\"|data_path: \"$TRAIN_PATHS_4D_STR\"|g" $TEMP_CONFIG
    sed -i "s|val_data_path: \".*\"|val_data_path: \"$VAL_PATH_4D\"|g" $TEMP_CONFIG
    sed -i "s|data_path2D: \".*\"|data_path2D: \"$TRAIN_PATHS_2D_STR\"|g" $TEMP_CONFIG
    sed -i "s|val_data_path2D: \".*\"|val_data_path2D: \"$VAL_PATH_2D\"|g" $TEMP_CONFIG
    # sed -i "s|data_path1D: \".*\"|data_path1D: \"$PATH_1D\"|g" $TEMP_CONFIG 

    # 调用训练脚本并传递临时配置文件
    
    # python $TRAIN_SCRIPT --config-name $TEMP_CONFIG
    python $TRAIN_SCRIPT 
done

echo "交叉验证完成"

# echo "Cross-validation Complete!"


