TensorFlowSharp are .NET bindings to the TensorFlow library published here:

https://github.com/tensorflow/tensorflow

This surfaces the C API as a strongly-typed C# API.

Work in progress - more details will come soon.

# Work in Progress

These instructions reflect what you need to get up and running with the
current code as I am working on it.   In the long-term, we will just have
NuGet packages that eliminate all the manual steps required here.

## Building your own version

You will want to install Tensorflow from sources, follow the instructions
for your platform here:

https://www.tensorflow.org/get_started/os_setup#installing_from_sources

This includes checking out the Tensorflow sources, installing Bazel, 
and building the core.

Once you do that, you will need to build the shared library, I believe
this is the command I used:

    bazel build -c opt //tensorflow:libtensorflow.so

If you want debug symbols for Tensorflow, while debugging the binding:

    bazel build -c dbg --strip=never //tensorflow:libtensorflow.so

You will need this library to be installed in a system accessible location
like /usr/local/lib, or in the directory of the application that you are
debugging.

## Running the test

I am currently using Xamarin Studio on a Mac to do the development, but this
should work on Windows with VS and Linux with MonoDevelop, there is nothing
Xamarin specific here.

Before the solution will run you will need the shared library generated to
be on the `SampleTest/bin/Debug directory`.   While Tensorflow builds a library
with the extension .so, you will need to make sure that it has the proper
name for your platform (tensorflow.dll on Windows, tensorflow.dylib on Mac)
and copy that there.

Tensorflow is a 64-bit library, so you will need to use a 64-bit Mono to run,
at home (where I am doing this work), I have a copy of 64-bit Mono on /mono,
so you will want to set that in your project configuration, to do this:

Open the project options (double click on the "SampleTest" project), then
select Run/Default, go to the "Advanced" tab, and select "Execute in .NET runtime"
and make sure that you select one that is 64-bit enabled.

Open the solution file in the top directory, and when you hit run, this will
run the API test.   

## Notes on OpDefs

Look at:

./tensorflow/core/ops/ops.pbtxt AvgPool3D and:
./tensorflow/core/ops/nn_ops.cc for the C++ implementation with type definitions

Docs on types:
https://www.tensorflow.org/versions/r0.11/how_tos/adding_an_op/

## Operation definitions:

Reference, high-level Go defintiions for OpDefs:

tensorflow/tensorflow/go/graph.go

wrappers.go generated from the build shows what our OpDef generation should look
like, I have not found the C++ equivalent generated code yet.
