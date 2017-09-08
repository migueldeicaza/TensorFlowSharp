#!/bin/sh
cd $TRAVIS_BUILD_DIR/packages/xunit.runner.console.2.2.0/tools
mono --arch=64 xunit.console.exe "$TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Debug/TensorFlowSharp.Tests.dll" "$TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests.CSharp/bin/Debug/TensorFlowSharp.Tests.CSharp.dll"