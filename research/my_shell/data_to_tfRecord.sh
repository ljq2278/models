python /home/ljq/tfOfficialModels/research/object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=/home/ljq/tfOfficialModels/research/object_detection/data/pascal_label_map.pbtxt \
    --data_dir=/media/ljq/Elements/dataset/voc2012/VOCdevkit --year=VOC2012 --set=train \
    --output_path=/home/ljq/tfOfficialModels/research/my_data/pascal_train.record
python /home/ljq/tfOfficialModels/research/object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=/home/ljq/tfOfficialModels/research/object_detection/data/pascal_label_map.pbtxt \
    --data_dir=/media/ljq/Elements/dataset/voc2012/VOCdevkit --year=VOC2012 --set=val \
    --output_path=/home/ljq/tfOfficialModels/research/my_data/pascal_val.record