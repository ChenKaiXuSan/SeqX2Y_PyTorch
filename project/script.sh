# #!/bin/bash

# # 定义数据集基本路径
# BASE_PATH="/workspace/data/579"

# # 定义训练脚本的路径
# TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/main.py"

# # 定义配置文件的原始路径和临时路径
# CONFIG_TEMPLATE="/workspace/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
# TEMP_CONFIG="/workspace/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# # 总共有12个患者的数据
# NUM_PATIENTS=12

# for i in $(seq 1 $NUM_PATIENTS)
# do
#     echo "开始第 $i 折交叉验证"

#     # 复制模板配置文件到临时配置文件
#     cp $CONFIG_TEMPLATE $TEMP_CONFIG

#     # 生成4DCT和2D和1D数据路径
#     TRAIN_PATHS_4D=()
#     TRAIN_PATHS_2D=()
#     VAL_PATH_4D="$BASE_PATH/POPI_seq3_579/4DCT-Dicom$i"
#     VAL_PATH_2D="$BASE_PATH/POPI_seq3_2D_579/2DCT-$i"
#     PATH_1D="/workspace/data/Diagram_Coordinates/1D_rpm.csv"

#     for j in $(seq 1 $NUM_PATIENTS)
#     do
#         if [ $j -ne $i ]
#         then
#             TRAIN_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j")
#             TRAIN_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$j")
#         fi
#     done

#     # 把训练路径数组转换为以逗号分隔的字符串
#     TRAIN_PATHS_4D_STR=$(IFS=,; echo "${TRAIN_PATHS_4D[*]}")
#     TRAIN_PATHS_2D_STR=$(IFS=,; echo "${TRAIN_PATHS_2D[*]}")

#     # 更新临时配置文件
#     sed -i "s|data_path: \".*\"|data_path: \"$TRAIN_PATHS_4D_STR\"|g" $TEMP_CONFIG
#     sed -i "s|val_data_path: \".*\"|val_data_path: \"$VAL_PATH_4D\"|g" $TEMP_CONFIG
#     sed -i "s|data_path2D: \".*\"|data_path2D: \"$TRAIN_PATHS_2D_STR\"|g" $TEMP_CONFIG
#     sed -i "s|val_data_path2D: \".*\"|val_data_path2D: \"$VAL_PATH_2D\"|g" $TEMP_CONFIG
#     sed -i "s|data_path1D: \".*\"|data_path1D: \"$PATH_1D\"|g" $TEMP_CONFIG 

#     # 调用训练脚本并传递临时配置文件
    
#     # python $TRAIN_SCRIPT --config-name $TEMP_CONFIG
#     python $TRAIN_SCRIPT 
# done

# echo "交叉验证完成"




#!/bin/bash
# 定义数据集基本路径
BASE_PATH="/workspace/data/579"

# 定义训练脚本的路径
TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/main.py"

# 定义配置文件的原始路径和临时路径
CONFIG_TEMPLATE="/workspace/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
TEMP_CONFIG="/workspace/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# 总共有12个患者的数据，每折包含2个患者
NUM_PATIENTS=12
NUM_FOLDS=6

for ((i=1; i<=NUM_FOLDS; i++))
do
    echo "开始第 $i 折交叉验证"

    # 复制模板配置文件到临时配置文件
    cp $CONFIG_TEMPLATE $TEMP_CONFIG

    # 计算验证集的患者序号
    VAL_START=$(( (i - 1) * 2 + 1 ))
    VAL_END=$(( VAL_START + 1 ))

    # 生成4DCT和2D和1D数据路径
    TRAIN_PATHS_4D=()
    TRAIN_PATHS_2D=()
    VAL_PATHS_4D=()
    VAL_PATHS_2D=()
    PATH_1D="/workspace/data/Diagram_Coordinates/1D_rpm.csv"

    for ((j=1; j<=NUM_PATIENTS; j++))
    do
        if [ $j -ge $VAL_START ] && [ $j -le $VAL_END ]
        then
            VAL_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j")
            VAL_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$j")
        else
            TRAIN_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j")
            TRAIN_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$j")
        fi
    done

    # 把训练路径和验证路径数组转换为以逗号分隔的字符串
    TRAIN_PATHS_4D_STR=$(IFS=,; echo "${TRAIN_PATHS_4D[*]}")
    TRAIN_PATHS_2D_STR=$(IFS=,; echo "${TRAIN_PATHS_2D[*]}")
    VAL_PATHS_4D_STR=$(IFS=,; echo "${VAL_PATHS_4D[*]}")
    VAL_PATHS_2D_STR=$(IFS=,; echo "${VAL_PATHS_2D[*]}")

    # 更新临时配置文件
    sed -i "s|data_path: \".*\"|data_path: \"$TRAIN_PATHS_4D_STR\"|g" $TEMP_CONFIG
    sed -i "s|val_data_path: \".*\"|val_data_path: \"$VAL_PATHS_4D_STR\"|g" $TEMP_CONFIG
    sed -i "s|data_path2D: \".*\"|data_path2D: \"$TRAIN_PATHS_2D_STR\"|g" $TEMP_CONFIG
    sed -i "s|val_data_path2D: \".*\"|val_data_path2D: \"$VAL_PATHS_2D_STR\"|g" $TEMP_CONFIG
    sed -i "s|data_path1D: \".*\"|data_path1D: \"$PATH_1D\"|g" $TEMP_CONFIG 

    # 调用训练脚本并传递临时配置文件
    python $TRAIN_SCRIPT
done

echo "交叉验证完成"