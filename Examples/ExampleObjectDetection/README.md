# Object Detection Example #

This example uses tensorflow [object detection model API](https://github.com/tensorflow/models/tree/master/object_detection) and TensorFlowSharp library to identify multiple objects in a single image using .NET programming languages like C# and F#.

![alt tag](demo-picture.jpg)

## Run example ##
1. ``` git clone https://github.com/migueldeicaza/TensorFlowSharp ```
2. build _TensorFlowSharp.sln_
3. copy _'libtensorflow.dylib'_ (Mac OS) or _'libtensorflow.dll'_ (Windows) to the project output path  (see where you can get the library under [Working on TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp#working-on-tensorflowsharp) section)
4. Run the ExampleObjectDetection util from command line.
```
ExampleObjectDetection
```

By default the example download pretrained model, but you can specify your own using following options:
_input_image_ - the path to the image for processing
_output_image_ - the path where the image with detected objects will be saved
_catalog_ - the path to the '*.pbtxt' file
_model_ - the path to the '*.pb' file 
 
for instance, 
```
ExampleObjectDetection --input_image="/demo/input.jpg" --output_image="/demo/output.jpg" --catalog="/demo/mscoco_label_map.pbtxt" --model="/demo/frozen_inference_graph.pb"
```

## I found an issue in the example ##
If you want to address a bug or a question related to the object detection example - just create a new issue on github starting with [Object Detection Example] tag.