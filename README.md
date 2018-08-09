[![Build Status](https://travis-ci.org/migueldeicaza/TensorFlowSharp.svg?branch=master)](https://travis-ci.org/migueldeicaza/TensorFlowSharp)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/TensorFlowSharp)

TensorFlowSharp are .NET bindings to the TensorFlow library published here:

https://github.com/tensorflow/tensorflow

This surfaces the C API as a strongly-typed .NET API for use from C# and F#.

The API surfaces the entire low-level TensorFlow API, it is on par with other
language bindings.  But currently does not include a high-level API like
the Python binding does, so it is more cumbersome to use for those high level
operations.

You can prototype using TensorFlow or Keras in Python, then save your graphs
or trained models and then load the result in .NET with TensorFlowSharp and
feed your own data to train or run.

The [current API
documentation](https://migueldeicaza.github.io/TensorFlowSharp/) is here.

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

Alternatively, you can [download it](https://www.nuget.org/packages/TensorFlowSharp/) directly.

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
    var addingResultValue = addingResults.GetValue();
    Console.WriteLine("a+b={0}", addingResultValue);

    // Multiply two constants
    var multiplyResults = session.GetRunner().Run(graph.Mul(a, b));
    var multiplyResultValue = multiplyResults.GetValue();
    Console.WriteLine("a*b={0}", multiplyResultValue);
}
```

Here is an F# scripting version of the same example, you can use this in F# Interactive:

```fsharp
#r @"packages\TensorFlowSharp.1.4.0\lib\net471\TensorFlowSharp.dll"

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

If you want to work on extending TensorFlowSharp or contribute to its development
read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

Please keep in mind that this requires a modern version of C# as this uses some
new capabilities there.   So you will want to use Visual Studio 2017.

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

## Documentation

Much of the online documentation comes from TensorFlow and is licensed under
the terms of Apache 2 License, in particular all the generated documentation
for the various operations that is generated by using the tensorflow reflection
APIs.

Last API update: Release 1.9
