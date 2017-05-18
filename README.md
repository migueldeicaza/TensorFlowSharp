[![Build Status](https://travis-ci.org/migueldeicaza/TensorFlowSharp.svg?branch=master)](https://travis-ci.org/migueldeicaza/TensorFlowSharp)

TensorFlowSharp are .NET bindings to the TensorFlow library published here:

https://github.com/tensorflow/tensorflow

This surfaces the C API as a strongly-typed C# API.

The API binding is pretty much done, and at this point, I am polishing the
API to make it more pleasant to use from C# and F# and resolving some of the
kinks and TODO-items that I left while I was doing the work.

My work-in-progress API documentation [current API
documentation](https://migueldeicaza.github.io/TensorFlowSharp/).

# Getting Started

You can either use the TensorFlow C-library release binaries, or build your own
from source.

- Linux
  - CPU-only: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.1.0.tar.gz
  - GPU-enabled: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.1.0.tar.gz
- Mac: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.1.0.tar.gz
- Windows: https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.2.0-rc0.zip

Unpack the above .tar.gz suitable for your system on a prefix that your
system's dynamic linker can use, for example, go to `/usr/local` and unpack there.

Mac note: the package contains a `.so` file, you will need to rename this to `.dylib` for
it to work.

Once you do that, you need to open the solution file on the top
level directory and build.   This will produce both the TensorFlowSharp
library as well as compile the tests and samples.

# Work in Progress

These instructions reflect what you need to get up and running with the
current code as I am working on it.   In the long-term, we will just have
NuGet packages that eliminate all the manual steps required here.

## Building your own version

To build the TensorFlow C library from source,
[try this](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md#building-the-tensorflow-c-library-from-source).

This includes checking out the Tensorflow sources, installing Bazel, 
and building the core.

Once you do that, you will need to build the shared library.
First, in the tensorflow directory, run:

    ./configure    

and answer the various prompts about your build. Important:
building with CUDA support provides better runtime performance
but has additional dependencies as discussed in the Tensorflow
installation Web page.

Once configured, run: 

    bazel build -c opt //tensorflow:libtensorflow.so

If you want debug symbols for Tensorflow, while debugging the binding:

    bazel build -c dbg --strip=never //tensorflow:libtensorflow.so

You will need the generated library (`libtensorflow.so`) to be installed in a
system accessible location like `/usr/local/lib`

On Linux:

```
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib/
```

On MacOS:

```
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib/libtensorflow.dylib
```

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
https://www.tensorflow.org/extend/adding_an_op

## Documentation

Much of the online documentation comes from TensorFlow and is licensed under
the terms of Apache 2 License, in particular all the generated documentation
for the various operations that is generated by using the tensorflow reflection
APIs.

Last API update: a4b352bfddd518b540c30e456f3bc0027ba9351f
