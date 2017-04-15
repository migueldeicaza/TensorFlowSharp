#!/bin/sh
nuget restore TensorFlowSharp.sln
cd $TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Release
mono xunit.console.exe TensorFlowSharp.Tests.dll xunit.console.exe.config