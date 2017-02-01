An example for using the TensorFlow F# API for image recognition
using a pre-trained inception model (http://arxiv.org/abs/1512.00567).

Sample usage: 

```
mono ExampleInceptionInference.exe [--dir=/tmp/modeldir] imagefile...

The pre-trained model takes input in the form of a 4-dimensional
tensor with shape [ BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3 ],
where:

- BATCH_SIZE allows for inference of multiple images in one pass through the graph
- IMAGE_HEIGHT is the height of the images on which the model was trained
- IMAGE_WIDTH is the width of the images on which the model was trained
- 3 is the (R, G, B) values of the pixel colors represented as a float.

And produces as output a vector with shape [ NUM_LABELS ].
output[i] is the probability that the input image was recognized as
having the i-th label.

A separate file contains a list of string labels corresponding to the
integer indices of the output.

This example:
- Loads the serialized representation of the pre-trained model into a Graph
- Creates a Session to execute operations on the Graph
- Converts an image file to a Tensor to provide as input to a Session run
- Executes the Session and prints out the label with the highest probability

To convert an image file to a Tensor suitable for input to the Inception model,
this example:
- Constructs another TensorFlow graph to normalize the image into a
  form suitable for the model (for example, resizing the image)
- Creates an executes a Session to obtain a Tensor in this normalized form.