# 30人随机交叉验证
#!/bin/bash
# 定义数据集基本路径
BASE_PATH="/workspace/data/579"

# 定义训练脚本的路径
TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/main.py"

# 定义配置文件的原始路径和临时路径
CONFIG_TEMPLATE="/workspace/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
TEMP_CONFIG="/workspace/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# 总共有30个患者的数据，每折包含5个患者
NUM_PATIENTS=30
NUM_FOLDS=6

# 生成患者ID列表
patient_ids=($(seq 1 $NUM_PATIENTS))
# 打乱患者ID列表
shuffled_ids=($(shuf -e "${patient_ids[@]}"))

for ((i=1; i<=NUM_FOLDS; i++))
do
    echo "开始第 $i 折交叉验证"

    # 复制模板配置文件到临时配置文件
    cp $CONFIG_TEMPLATE $TEMP_CONFIG

    # 计算验证集的患者序号
    VAL_START=$(( (i - 1) * 5 ))
    VAL_END=$(( VAL_START + 4 ))

    # 生成4DCT和2D和1D数据路径
    TRAIN_PATHS_4D=()
    TRAIN_PATHS_2D=()
    VAL_PATHS_4D=()
    VAL_PATHS_2D=()
    PATH_1D="/workspace/data/Diagram_Coordinates/1D_rpm.csv"

    for ((j=0; j<$NUM_PATIENTS; j++))
    do
        patient_id=${shuffled_ids[$j]}
        if [ $j -ge $VAL_START ] && [ $j -le $VAL_END ]
        then
            VAL_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$patient_id")
            VAL_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$patient_id")
        else
            TRAIN_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$patient_id")
            TRAIN_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$patient_id")
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



# 30人，6折交叉验证
# #!/bin/bash

# # 定义数据集基本路径
# BASE_PATH="/workspace/data/579"

# # 定义训练脚本的路径
# TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/main.py"

# # 定义配置文件的原始路径和临时路径
# CONFIG_TEMPLATE="/workspace/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
# TEMP_CONFIG="/workspace/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# # 总共有30个患者的数据，每折包含5个患者
# NUM_PATIENTS=30
# NUM_FOLDS=6

# for ((i=1; i<=NUM_FOLDS; i++))
# do
#     echo "开始第 $i 折交叉验证"

#     # 复制模板配置文件到临时配置文件
#     cp $CONFIG_TEMPLATE $TEMP_CONFIG

#     # 计算验证集的患者序号
#     VAL_START=$(( (i - 1) * 5 + 1 ))
#     VAL_END=$(( VAL_START + 4 ))

#     # 生成4DCT和2D和1D数据路径
#     TRAIN_PATHS_4D=()
#     TRAIN_PATHS_2D=()
#     VAL_PATHS_4D=()
#     VAL_PATHS_2D=()
#     PATH_1D="/workspace/data/Diagram_Coordinates/1D_rpm.csv"

#     for ((j=1; j<=NUM_PATIENTS; j++))
#     do
#         if [ $j -ge $VAL_START ] && [ $j -le $VAL_END ]
#         then
#             VAL_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j")
#             VAL_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$j")
#         else
#             TRAIN_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j")
#             TRAIN_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$j")
#         fi
#     done

#     # 把训练路径和验证路径数组转换为以逗号分隔的字符串
#     TRAIN_PATHS_4D_STR=$(IFS=,; echo "${TRAIN_PATHS_4D[*]}")
#     TRAIN_PATHS_2D_STR=$(IFS=,; echo "${TRAIN_PATHS_2D[*]}")
#     VAL_PATHS_4D_STR=$(IFS=,; echo "${VAL_PATHS_4D[*]}")
#     VAL_PATHS_2D_STR=$(IFS=,; echo "${VAL_PATHS_2D[*]}")

#     # 更新临时配置文件
#     sed -i "s|data_path: \".*\"|data_path: \"$TRAIN_PATHS_4D_STR\"|g" $TEMP_CONFIG
#     sed -i "s|val_data_path: \".*\"|val_data_path: \"$VAL_PATHS_4D_STR\"|g" $TEMP_CONFIG
#     sed -i "s|data_path2D: \".*\"|data_path2D: \"$TRAIN_PATHS_2D_STR\"|g" $TEMP_CONFIG
#     sed -i "s|val_data_path2D: \".*\"|val_data_path2D: \"$VAL_PATHS_2D_STR\"|g" $TEMP_CONFIG
#     sed -i "s|data_path1D: \".*\"|data_path1D: \"$PATH_1D\"|g" $TEMP_CONFIG 

#     # 调用训练脚本并传递临时配置文件
#     python $TRAIN_SCRIPT
# done

# echo "交叉验证完成"





# # 留一验证，计算量最大！
# #!/bin/bash
# # 定义数据集基本路径
# BASE_PATH="/workspace/data/579"

# # 定义训练脚本的路径
# TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/main.py"

# # 定义配置文件的原始路径和临时路径
# CONFIG_TEMPLATE="/workspace/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
# TEMP_CONFIG="/workspace/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# # 总共有30个患者的数据
# NUM_PATIENTS=30

# for i in $(seq 1 $NUM_PATIENTS)
# do
#     echo "开始第 $i 折交叉验证"

#     # 复制模板配置文件到临时配置文件
#     cp $CONFIG_TEMPLATE $TEMP_CONFIG

#     # 生成4DCT和2D数据路径
#     TRAIN_PATHS_4D=()
#     TRAIN_PATHS_2D=()
#     VAL_PATH_4D="$BASE_PATH/POPI_seq3_579/4DCT-Dicom$i"
#     VAL_PATH_2D="$BASE_PATH/POPI_seq3_2D_579/2DCT-$i"

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

#     # 调用训练脚本并传递临时配置文件
    
#     # python $TRAIN_SCRIPT --config-name $TEMP_CONFIG
#     python $TRAIN_SCRIPT 
# done

# echo "交叉验证完成"





# 第二版
# #!/bin/bash

# # 定义数据集基本路径
# BASE_PATH="/workspace/data/579"

# # 定义训练脚本的路径
# TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/train.py"

# # 定义配置文件的原始路径和临时路径
# CONFIG_TEMPLATE="/workspace/SeqX2Y_PyTorch/configs/data/4DCT.yaml"
# TEMP_CONFIG="/workspace/SeqX2Y_PyTorch/configs/data/Temp_4DCT.yaml"

# # 总共有6个患者的数据
# NUM_PATIENTS=6

# for i in $(seq 1 $NUM_PATIENTS)
# do
#     echo "Start the $i fold Cross-validation"

#     # 复制模板配置文件到临时配置文件
#     cp $CONFIG_TEMPLATE $TEMP_CONFIG

#     # 为每一个患者生成训练集和测试集路径，并更新配置文件
#     for j in $(seq 1 $NUM_PATIENTS)
#     do
#         if [ $j -eq $i ]
#         then
#             sed -i "s|data_path: \".*\"|data_path: \"$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j\"|g" $TEMP_CONFIG
#             sed -i "s|val_data_path: \".*\"|val_data_path: \"$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j\"|g" $TEMP_CONFIG
#             sed -i "s|data_path2D: \".*\"|data_path2D: \"$BASE_PATH/POPI_seq3_2D_579/2DCT-$j\"|g" $TEMP_CONFIG
#             sed -i "s|val_data_path2D: \".*\"|val_data_path2D: \"$BASE_PATH/POPI_seq3_2D_579/2DCT-$j\"|g" $TEMP_CONFIG
#         fi
#     done

#     # 调用训练脚本并传递临时配置文件
#     python $TRAIN_SCRIPT --config $TEMP_CONFIG
# done

# echo "Cross-validation Complete!"

#======================================================================

# 初版
# #!/bin/bash

# # 定义数据集基本路径
# BASE_PATH="/workspace/data/579"

# # 定义训练脚本的路径
# TRAIN_SCRIPT="/workspace/SeqX2Y_PyTorch/project/train.py"

# # 总共有6个患者的数据
# NUM_PATIENTS=6

# for i in $(seq 1 $NUM_PATIENTS)
# do
#     echo "Start the $i fold Cross-validation"

#     # 生成4DCT训练集和测试集路径
#     TRAIN_PATHS_4D=()
#     VAL_PATH_4D=""

#     # 生成2D训练集和测试集路径
#     TRAIN_PATHS_2D=()
#     VAL_PATH_2D=""

#     for j in $(seq 1 $NUM_PATIENTS)
#     do
#         if [ $j -eq $i ]
#         then
#             VAL_PATH_4D="$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j"
#             VAL_PATH_2D="$BASE_PATH/POPI_seq3_2D_579/2DCT-$j"
#         else
#             TRAIN_PATHS_4D+=("$BASE_PATH/POPI_seq3_579/4DCT-Dicom$j")
#             TRAIN_PATHS_2D+=("$BASE_PATH/POPI_seq3_2D_579/2DCT-$j")
#         fi
#     done

#     # 把训练路径数组转换为以空格分隔的字符串
#     TRAIN_PATHS_4D_STR=$(IFS=" "; echo "${TRAIN_PATHS_4D[*]}")
#     TRAIN_PATHS_2D_STR=$(IFS=" "; echo "${TRAIN_PATHS_2D[*]}")

#     # 调用训练脚本
#     python $TRAIN_SCRIPT --data_path "$TRAIN_PATHS_4D_STR" --val_data_path "$VAL_PATH_4D" --data_path2D "$TRAIN_PATHS_2D_STR" --val_data_path2D "$VAL_PATH_2D"
# done

# echo "Cross-validation Complete!"


