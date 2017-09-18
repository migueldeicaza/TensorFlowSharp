

if [[ ! -e 'bin/Debug/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017' ]]; then
wget "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz"
    tar -zxvf faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz -C bin/Debug/
elif [[ ! -d $dir ]]; then
    echo "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017 already exists"
fi


cp -R test_images ./bin/Debug/test_images
cd bin/Debug
wget "https://raw.githubusercontent.com/tensorflow/models/master/object_detection/data/mscoco_label_map.pbtxt"
mono --arch=64 ExampleObjectDetection.exe --input_image=$(pwd)/test_images/input.jpg --output_image=$(pwd)/test_images/output.jpg --catalog=$(pwd)/mscoco_label_map.pbtxt --model=$(pwd)/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb
echo "detection completed"
