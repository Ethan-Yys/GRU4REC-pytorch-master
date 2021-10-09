opt=$1
YMD=$2

if [ ! $opt ]
then
  opt="tp"
fi

if [ ! $YMD ]
then
  #YMD=$(date +%Y%m%d -u)
  YMD=$(date -d "-1 day" +%Y%m%d)
fi
YMD_1D_AGO=$(date -d"yesterday ${YMD}" +%Y%m%d)
# 加载shell tools辅助函数
source spark_util/shell_tools.sh


function check_cmd(){
    if [ $1 -ne 0 ];then
        exit $1
    fi
}


activate_dir='/home/ec2-user/anaconda3/bin/activate'
source ${activate_dir} py37_torch

data_base="s3://mx-machine-learning/yuyisong/takatak/user_act_daily_intermediate/${YMD}"
model_upload_path="s3://mx-machine-learning/yuyisong/takatak/gru4rec_model"
wait_s3_data ${data_base} 18
check_cmd $?

if [[ $opt == *t* ]]
then
    # 训练阶段

    if [ ! -e model_saved ]
    then
        mkdir model_saved
    fi

    if [ ! -e feature_data ]
    then
        mkdir feature_data
    else
        rm -r feature_data
        mkdir feature_data
    fi



    aws s3 cp ${data_base} feature_data --recursive --quiet
    #gzip -d feature_data/train/*.gz
    #gzip -d feature_data/test/*.gz

    python gru_main.py --checkpoint_dir model_saved --model_file_name gru4rec_${YMD}.pt --train_data ./feature_data/*.csv.gz
    check_cmd $?


    aws s3 cp model_saved/gru4rec_${YMD}.pt "${model_upload_path}/gru4rec_${YMD}.pt" --quiet

fi


if [[ $opt == *p* ]]
then

    wait_s3_file "${model_upload_path}/gru4rec_${YMD}.pt" 24
    check_cmd $?

    if [ ! -e model_saved ]
    then
        mkdir model_saved
        aws s3 cp "${model_upload_path}/gru4rec_${YMD}.pt" model_saved/gru4rec_${YMD}.pt  --quiet
    fi


    if [ ! -e predict_output ]
    then
        mkdir predict_output
    fi

    # gzip -d predict_data/*.gz

    python gru_main.py --checkpoint_dir model_saved --model_file_name gru4rec_${YMD}.pt --is_eval --load_model model_saved/gru4rec_${YMD}.pt --train_data ./feature_data/*.csv.gz

    check_cmd $?

    predict_upload="s3://mx-machine-learning/yuyisong/takatak/predict_output/${YMD}"

    aws s3 cp predict_output/ ${predict_upload} --recursive --quiet
    check_cmd $?

    touch predict_output/_SUCCESS
    aws s3 cp predict_output/_SUCCESS "${predict_upload}/_SUCCESS" --quiet

    rm -r model_saved/
fi