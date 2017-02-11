TensorFlowSharp are .NET bindings to the TensorFlow library published here:

https://github.com/tensorflow/tensorflow

This surfaces the C API as a strongly-typed C# API.

The API binding is pretty much done, and at this point, I am polishing the
API to make it more pleasant to use from C# and F# and resolving some of the
kinks and TODO-items that I left while I was doing the work.

# Getting Started

You need to get yourself a copy of the TensorFlow runtime, you can either
build your own version (recommended, see the instructions below) or you can use a precompiled
binary:

https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.0.0-rc0.tar.gz
https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-darwin-x86_64-1.0.0-rc0.tar.gz
https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.0.0-rc0.tar.gz
https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.0.0-rc0.tar.gz

Unpack the above .tar.gz suitable for your system on a prefix that your
system's dynamic linker can use, for example, go to `/usr/local` and unpack there.

Mac note: the package contains a `.so` file, you will need to rename this to `.dylib` for
it to work.

Once you do that, you need to open the solution file on the top
level directory and build.   This will produce both the TensorFlowSharp
library as well as compile the tests and samples.

It is recommended that you build your own, because these bindings of TensorFlow surface some
features in the latest version of TensorFlow that are not available on the 1.0.0-rc0 builds above.

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
be on a location accessibly by the Mono runtime (for example /usr/local/lib).

While Tensorflow builds a library with the extension .so, you will need 
to make sure that it has the proper name for your platform (tensorflow.dll on Windows, 
tensorflow.dylib on Mac) and copy that there.

Tensorflow is a 64-bit library, so you will need to use a 64-bit Mono to run,
at home (where I am doing this work), I have a copy of 64-bit Mono on /mono,
so you will want to set that in your project configuration, to do this:

Open the project options (double click on the "SampleTest" project), then
select Run/Default, go to the "Advanced" tab, and select "Execute in .NET runtime"
and make sure that you select one that is 64-bit enabled.

Open the solution file in the top directory, and when you hit run, this will
run the API test.   

## Possible Contributions

### Build More Tests

Would love to have more tests to ensure the proper operation of the framework.

### Samples

The binding is pretty much complete, and at this point, I want to improve the 
API to be easier and more pleasant to use from both C# and F#.   Creating
samples that use Tensorflow is a good way of finding easy wins on the usability
of the API, there are some here:

https://github.com/tensorflow/models

### Packaging

x86: It is not clear to me how to distribute the native libtensorflow to users, as
it is designed to be compiled for your host system.  I would like to figure out
how we can distribute packages that have been compiled with the optimal set of
optimizations for users to consume.

Mobile: we need to package the library for consumption on Android and iOS.

### NuGet Package

Would love to have a NuGet package for all platforms.

### Issues

I have logged some usability problems and bugs in Issues, feel free to take
on one of those tasks.

## Notes on OpDefs

Look at:

./tensorflow/core/ops/ops.pbtxt AvgPool3D and:
./tensorflow/core/ops/nn_ops.cc for the C++ implementation with type definitions

Docs on types:
https://www.tensorflow.org/versions/r0.11/how_tos/adding_an_op/


