# Getting Started With TensorFlowSharp

TensorFlowSharp provides APIs for use in .NET programs, including C#
and F#. These APIs are particularly well-suited to loading models
created in Python and executing them within a .NET application. This
guide explains how to install TensorFlow for .NET and use it in your application.

TensorFlowSharp itself is a .NET API that calls into the native
TensorFlow runtime.   

# Supported Platforms

The NuGet package that you obtain from NuGet.org comes with the native
TensorFlow runtime for Windows (x64), Mac (x64) and Linux (x64).    

If you desire to run TensorFlowSharp in other platforms, you can do so
by downloading the appropriate TensorFlow dynamic library for your
platform and placing this side-by-side the `TensorFlowSharp.dll` library.

It is just not included by default as this would make the binary a lot larger.

Additionally, up until version 1.5, TensorFlowSharp currently ships
with .NET Desktop libraries that run on the .NET Desktop on Windows or
on Linux and Mac using the Mono runtime.

Support for running under .NET Core is waiting on the [migration of the
package to the .NET Standard](https://github.com/migueldeicaza/TensorFlowSharp/pull/188).

# Using TensorFlowSharp in a .NET Application

To use TensorFlowSharp, you will need to create a .NET Desktop
application on Windows or using Mono on Linux and Mac.   

To use it, make sure that you download the TensorFlowSharp package
from NuGet, either using the command line (`nuget install
TensorFlowSharp`) or from your favorite .NET IDE.

# Getting started


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

```
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

