INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/ljq/tfOfficialModels/research/my_config/detect_mobilenet/ssdlite_mobilenet_v2_coco.config
TRAINED_CKPT_PREFIX=/home/ljq/tfOfficialModels/research/my_model/detect_mobilenet/model.ckpt-0
EXPORT_DIR=/home/ljq/tfOfficialModels/research/my_export/detect_mobilenet
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
