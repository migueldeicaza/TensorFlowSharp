#!/bin/sh
cd $TRAVIS_BUILD_DIR/packages/xunit.runner.console.2.2.0/tools
mono --arch=64 xunit.console.exe "$TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Release/TensorFlowSharp.Tests.dll" "$TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests.CSharp/bin/Release/TensorFlowSharp.Tests.CSharp.dll"