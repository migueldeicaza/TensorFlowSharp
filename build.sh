#!/bin/sh
cd $TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Release
mono --arch=64 xunit.console.exe TensorFlowSharp.Tests.dll xunit.console.exe.config