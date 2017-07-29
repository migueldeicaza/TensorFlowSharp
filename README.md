[![Build Status](https://travis-ci.org/migueldeicaza/TensorFlowSharp.svg?branch=master)](https://travis-ci.org/migueldeicaza/TensorFlowSharp)

TensorFlowSharp are .NET bindings to the TensorFlow library published here:

https://github.com/tensorflow/tensorflow

This surfaces the C API as a strongly-typed .NET API for use from C# and F#.

The API binding is pretty much done, and at this point, I am polishing the
API to make it more pleasant to use from C# and F# and resolving some of the
kinks and TODO-items that I left while I was doing the work.

My work-in-progress API documentation [current API
documentation](https://migueldeicaza.github.io/TensorFlowSharp/).

# Using TensorFlowSharp

## Installation 

The easiest way to get started is to use the NuGet package for 
TensorFlowSharp which contains both the .NET API as well as the 
native libraries for 64-bit Linux, Mac and Windows using the CPU backend.

You can install using NuGet like this:

```cmd
nuget install TensorFlowSharp
```

Or select it from the NuGet packages UI on Visual Studio.


On Visual Studio, make sure that you are targeting .NET 4.6.1 or
later, as this package uses some features of newer .NETs.  Otherwise,
the package will not be added. Once you do this, you can just use the
TensorFlowSharp nuget

Alternatively, you can [download it](https://www.nuget.org/api/v2/package/TensorFlowSharp/1.2.2) directly.

## Using TensorFlowSharp

Your best source of information right now are the SampleTest that
exercises various APIs of TensorFlowSharp, or the stand-alone samples
located in "Examples".

This API binding is closer design-wise to the Java and Go bindings
which use explicit TensorFlow graphs and sessions.  Your application
will typically create a graph (TFGraph) and setup the operations
there, then create a session from it (TFSession), then use the session
runner to setup inputs and outputs and execute the pipeline.

Something like this:

```csharp
using(var graph = new TFGraph ())
{
    graph.Import (File.ReadAllBytes ("MySavedModel"));
    var session = new TFSession (graph);
    var runner = session.GetRunner ();
    runner.AddInput (graph ["input"] [0], tensor);
    runner.Fetch (graph ["output"] [0]);

    var output = runner.Run ();

    // Fetch the results from output:
    TFTensor result = output [0];
}
```

In scenarios where you do not need to setup the graph independently,
the session will create one for you.  The following example shows how
to abuse TensorFlow to compute the addition of two numbers:

```csharp
using (var session = new TFSession())
{
    var graph = session.Graph;

    var a = graph.Const(2);
    var b = graph.Const(3);
    Console.WriteLine("a=2 b=3");

    // Add two constants
    var addingResults = session.GetRunner().Run(graph.Add(a, b));
    var addingResultValue = addingResults[0].GetValue();
    Console.WriteLine("a+b={0}", addingResultValue);

    // Multiply two constants
    var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
    var multiplyResultValue = multiplyResults.GetValue();
    Console.WriteLine("a*b={0}", multiplyResultValue);
}
```
Here is an F# scripting version of the same example, you can use this in F# Interactive:
```
#r @"packages\TensorFlowSharp.1.2.2\lib\net461\TensorFlowSharp.dll"

open System
open System.IO
open TensorFlow

// set the path to find the native DLL
Environment.SetEnvironmentVariable("Path", 
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__ + @"/packages/TensorFlowSharp.1.2.2/native")

module AddTwoNumbers = 
    let session = new TFSession()
    let graph = session.Graph

    let a = graph.Const(new TFTensor(2))
    let b = graph.Const(new TFTensor(3))
    Console.WriteLine("a=2 b=3")

    // Add two constants
    let addingResults = session.GetRunner().Run(graph.Add(a, b))
    let addingResultValue = addingResults.GetValue()
    Console.WriteLine("a+b={0}", addingResultValue)

    // Multiply two constants
    let multiplyResults = session.GetRunner().Run(graph.Mul(a, b))
    let multiplyResultValue = multiplyResults.GetValue()
    Console.WriteLine("a*b={0}", multiplyResultValue)
```


# Working on TensorFlowSharp 

TensorFlowSharp are bindings to the native TensorFlow library.

You can either use the TensorFlow C-library release binaries, or build
your own from source.  Here are some pre-built TensorFlow binaries you
can use for each platform:

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

## Building your own native TensorFlow library

To build the TensorFlow C library from source,
[follow these instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md#building-the-tensorflow-c-library-from-source).

This includes checking out the Tensorflow sources, installing Bazel, 
and building the core.

Once you do that, you will need to build the shared library.
First, in the tensorflow directory, run:

```bash
./configure    
```

and answer the various prompts about your build. Important:
building with CUDA support provides better runtime performance
but has additional dependencies as discussed in the Tensorflow
installation Web page.

Once configured, run: 

```bash
bazel build -c opt //tensorflow:libtensorflow.so
```

If you want debug symbols for Tensorflow, while debugging the binding:

```bash
bazel build -c dbg --strip=never //tensorflow:libtensorflow.so
```

You will need the generated library (`libtensorflow.so`) to be installed in a
system accessible location like `/usr/local/lib`

On Linux:

```bash
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib/
```

On MacOS:

```bash
sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib/libtensorflow.dylib
```

## Running the test

I am currently using Visual Studio for Mac to do the development, but this
should work on Windows with VS and Linux with MonoDevelop.

Before the solution will run you will need the shared library generated to
be on a location accessibly by the Mono runtime (for example /usr/local/lib).

While Tensorflow builds a library with the extension .so, you will need 
to make sure that it has the proper name for your platform (tensorflow.dll on Windows, 
tensorflow.dylib on Mac) and copy that there.

Tensorflow is a 64-bit library, so you will need to use a 64-bit Mono to run,
at home (where I am doing this work), I have a copy of 64-bit Mono on /mono,
so you will want to set that in your project configuration, to do this:

Ensure that your Build/Compiler settings set "Platform Target" to "x64".

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

Mobile: we need to package the library for consumption on Android and iOS.

### Documentation Styling

The API documentation has not been styled, I am using the barebones template
for documentation, and it can use some work.

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
