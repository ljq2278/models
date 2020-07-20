PIPELINE_CONFIG_PATH=/home/ljq/tfOfficialModels/research/my_config/detect_mobilenet/ssdlite_mobilenet_v2_coco.config
MODEL_DIR=/home/ljq/tfOfficialModels/research/my_model/detect_mobilenet
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --alsologtostderr
