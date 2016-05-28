BASE_DIR=$1
data_prefix=$2
run_id=$3
echo ${BASE_DIR}
echo ${data_prefix}
echo ${run_id}
start_time=$(date)
THEANO_FLAGS=device=gpu1,nvcc.fastmath=True,openmp=True python projects/deepdialog/chat/nn/main.py -d 125 -i 50 -o 50 -t 35 --batch-size 5 -c lstm -m encoderdecoder --train-data ${BASE_DIR}/${data_prefix}.train --dev-data ${BASE_DIR}/${data_prefix}.val --stats-file ${BASE_DIR}/${run_id}/eval_stats.out --save-file ${BASE_DIR}/${run_id}/model.out --dev-eval-file ${BASE_DIR}/${run_id}/dev_eval.out --train-eval-file ${BASE_DIR}/${run_id}/train_eval.out > ${BASE_DIR}/${run_id}/out.log
end_time=$(date)
echo ${start_time} >> ${BASE_DIR}/${run_id}/out.log
echo ${end_time} >> ${BASE_DIR}/${run_id}/out.log
