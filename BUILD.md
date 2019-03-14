# How to Build and Test

Unfortunately, the process is currently more manual than would be ideal.  If you want to build TensorFlowSharp
from source, you first need to get the TensorFlow native binaries.  CONTRIBUTING.md contains info about how to
build those from source as well or download pre-built binaries from Google.  Alternatively, you can extract the
pre-built binaries from the official TensorFlowSharp nuget package.  This is the easiest approach.  

To do that, browse to nuget.org, find the latest TensorFlowSharp package and download the `.nupkg` file.  Unzip
it (nupkg files are just zip files with a different extension) and copy the `runtimes` directory and all of its
subdirectories to `runtimes` at the root of this repo.

## Building

Once the native libraries have been copied to the `runtimes` directory, you should be able to build the project with
`dotnet build`.  This will build for .net4.5, .net4.6 and newer and netstandard2.0.  If you specify, `-c Release` as an
option on the build commandline, it will also produce a copy of the nupkg file with all three binary types packed inside
(as well as the native libraries for windows, osx and linux).

## Testing

In order to test on net45 or net471, you will need to download `xunit.console.exe`.  Once again, you can browse to
nuget.org and find the `2.1.0` version of the package `xunit.runners.console`.  You must use this old version of xunit
beacuse newer versions do not support net45.  Download the nupkg file, unzip it, and copy the `tools` directory to 
`tools` at the root of this repo.

### Testing for Net45

From the root of the repo run `tools\xunit.console.exe tests\TensorFlowSharp.Tests\bin\debug\net45\TensorFlowSharp.Tests.dll tests\TensorFlowSharp.Tests.CSharp\bin\debug\net45\TensorFlowSharp.Tests.CSharp.dll`

On windows 6 tests fail.  Results may be slightly different on other platforms.

### Testing for Net471

From the root of the repo run `tools\xunit.console.exe tests\TensorFlowSharp.Tests\bin\debug\net471\TensorFlowSharp.Tests.dll tests\TensorFlowSharp.Tests.CSharp\bin\debug\net471\TensorFlowSharp.Tests.CSharp.dll`

On windows 6 tests fail. Results may be slightly different on other platforms.

### Testing for NetStandard2.0

From the root of the repo run `dotnet test`.

On Windows 1 test fails in the f# test project, and 6 tests fail in the c# test project.  Results may be slightly
different on other platforms.