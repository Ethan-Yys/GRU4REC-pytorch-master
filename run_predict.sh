opt=$1
YMD=$2

if [ ! $opt ]
then
  opt="tp"
fi

if [ ! $YMD ]
then
  YMD=$(date +%Y%m%d -u)
fi
YMD_1D_AGO=$(date -d"yesterday ${YMD}" +%Y%m%d)
# 加载shell tools辅助函数
source ../../spark_util/shell_tools.sh


function check_cmd(){
    if [ $1 -ne 0 ];then
        exit $1
    fi
}

activate_dir='/home/ec2-user/anaconda3/bin/activate'
source ${activate_dir} py37_torch

if [[ $opt == *t* ]]
then
    # 训练阶段
    feature_base="s3://mx-machine-learning/dongxinzhou/takatak/v2/feature/${YMD}"
    wait_s3_file ${feature_base}/feature_stat 18
    check_cmd $?

    if [ ! -e model_saved_v2 ]
    then
        mkdir model_saved_v2
    fi

    model_upload_path="s3://mx-machine-learning/dongxinzhou/takatak/v2/din_model"
    aws s3 cp "${model_upload_path}/din_${YMD_1D_AGO}.pth" model_saved_v2/din_${YMD_1D_AGO}.pth  --quiet


    if [ ! -e feature_data_v2 ]
    then
        mkdir feature_data_v2
    else
        rm -r feature_data_v2
        mkdir feature_data_v2
    fi

    aws s3 cp ${feature_base} feature_data_v2 --recursive --quiet
    #gzip -d feature_data/train/*.gz
    #gzip -d feature_data/test/*.gz

    if [ -e model_saved_v2/din_${YMD_1D_AGO}.pth ]
    then
        python din_main.py --task train --train_input_path feature_data_v2/train/ --eval_input_path feature_data_v2/test/ \
                        --model_config model_config_v2.json --model_path model_saved_v2/din_${YMD}.pth --batch_size 256 \
                        --sampler_workers 4 --max_train_minutes 300 --pretrained_model_path model_saved_v2/din_${YMD_1D_AGO}.pth
        check_cmd $?
    else
        python din_main.py --task train --train_input_path feature_data_v2/train/ --eval_input_path feature_data_v2/test/ \
                        --model_config model_config_v2.json --model_path model_saved_v2/din_${YMD}.pth --batch_size 256 \
                        --sampler_workers 4 --max_train_minutes 300
        check_cmd $?
    fi

    aws s3 cp model_saved_v2/din_${YMD}.pth "${model_upload_path}/din_${YMD}.pth" --quiet

    rm -r feature_data_v2/
fi


if [[ $opt == *p* ]]
then
    rm -r predict_data_v2
    rm -r predict_output_v2
    # 预测阶段，等待数据
    predict_samples="s3://mx-machine-learning/libeibei/din_model/predict_v2/${YMD}"
    wait_s3_data ${predict_samples} 18
    check_cmd $?
    # 等待模型
    model_upload_path="s3://mx-machine-learning/dongxinzhou/takatak/v2/din_model"
    wait_s3_file "${model_upload_path}/din_${YMD}.pth" 24
    check_cmd $?



    if [ ! -e model_saved_v2 ]
    then
        mkdir model_saved_v2
    fi
    aws s3 cp "${model_upload_path}/din_${YMD}.pth" model_saved_v2/din_${YMD}.pth  --quiet

    if [ ! -e predict_data_v2 ]
    then
        mkdir predict_data_v2
    fi
    if [ ! -e predict_output_v2 ]
    then
        mkdir predict_output_v2
    fi
    aws s3 cp ${predict_samples} predict_data_v2 --recursive --quiet
    # gzip -d predict_data/*.gz

    python din_main.py --task predict --model_path model_saved_v2/din_${YMD}.pth --model_config model_config_v2.json \
                        --predict_input_path predict_data_v2/ --predict_output_path predict_output_v2/ --batch_size 1024

    check_cmd $?

    predict_upload="s3://mx-machine-learning/dongxinzhou/takatak/v2/predict_samples/${YMD}"
    gzip predict_output_v2/part-*
    aws s3 cp predict_output_v2/ ${predict_upload} --recursive --quiet
    check_cmd $?

    touch predict_output_v2/_SUCCESS
    aws s3 cp predict_output_v2/_SUCCESS "${predict_upload}/_SUCCESS" --quiet

    rm -r model_saved_v2/
fi