AWS_local_GardCam_SeqX2Y的readme，将可能会涉及到修改的py文件解释如下：

project
    models
        -Time_series_seq2seq_4DCT_voxelmorph.py 
            目前包含了Encoder3DCNN(处理时间序列2D图片的模型)，EncoderDecoderConvLSTM(处理时间序列4D CT的模型)
            需求：将Encoder3DCNN和EncoderDecoderConvLSTM拆开，能够满足将Encoder3DCNN提取的2D图片结果单独输入grad cam进行计算，并且能将各个2D图片结果按照时间序列保存(目前时间序列是3，按顺序输入3张2d图片，希望能将每张的grad cam结果都保存)
    
    utils
        -base_cam_2D.py
            原代码BaseCAM基类中调用了ActivationsAndGradients基类，由于需要额外输入batch_2D和future_seq，经过修改变BaseCAM_2D和ActivationsAndGradients_2D
        -grad_cam.py
            我自己写的GradCAM_2D(BaseCAM): 继承我自己写的BaseCAM_2D
        -image_saver.py
            保存训练后得到的预测4d ct和groundtruth
    
    -loss_analyst.py(原本loss的计算和评价指标计算是写在train.py里面的，由于太长了就单独写到了一个py文件里面)
        这里面进行评价指标ssim，psnr，mae的计算，还有train_loss(calculate_train_loss)，validation_loss(calculate_val_loss)的计算

    -train.py
        进行训练，只有train step和validation step，没有test step    

