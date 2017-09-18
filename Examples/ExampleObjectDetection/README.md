# Object Detection Example #

This example uses tensorflow [object detection model API](https://github.com/tensorflow/models/tree/master/object_detection) and TensorFlowSharp library to identify multiple objects in a single image using .NET programming languages like C# and F#.

![alt tag](demo-picture.jpg)

## Run example on Windows ##
1. ``` git clone https://github.com/migueldeicaza/TensorFlowSharp ```
2. build _TensorFlowSharp.sln_ in _Debug_ configuration
3. copy _'libtensorflow.dll'_ to _'TensorFlowSharp\Examples\ExampleObjectDetection\bin\Debug'_ folder (see where you can get the library under [Working on TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp#working-on-tensorflowsharp) section)
4. go to _'TensorFlowSharp\Examples\ExampleObjectDetection'_ folder and run _'run_example_windows.ps1'_ PowerShell script. This step detects objects in the _'TensorFlowSharp\Examples\ExampleObjectDetection\test_images\input.jpg'_ image and saves result to _'TensorFlowSharp\Examples\ExampleObjectDetection\bin\Debug\test_images\output.jpg'_ image.

## Run example on Mac OS ##
1. ```git clone https://github.com/migueldeicaza/TensorFlowSharp```
2. build _TensorFlowSharp.sln_ with _Debug_ configuration using,for instance, Visual Studio for Mac
3. copy _'libtensorflow.dylib'_ to _'TensorFlowSharp\Examples\ExampleObjectDetection\bin\Debug'_ folder  (see where you can get the library under [Working on TensorFlowSharp](https://github.com/migueldeicaza/TensorFlowSharp#working-on-tensorflowsharp) section)
4. go to _'TensorFlowSharp\Examples\ExampleObjectDetection'_ folder and run _'run_example_macos.sh'_. This step detects objects in the _'TensorFlowSharp\Examples\ExampleObjectDetection\test_images\input.jpg'_ image and saves result to _'TensorFlowSharp\Examples\ExampleObjectDetection\bin\Debug\test_images\output.jpg'_ image.

## I found an issue in the example ##
If you want to address a bug or a question related to the object detection example, just create new issue on github starting with [Object Detection Example] tag.