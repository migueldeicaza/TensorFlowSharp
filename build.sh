#!/bin/sh
cp -R $TRAVIS_BUILD_DIR/release-artifacts/libs/libtensorflow.dylib $TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Release/
cp -R $TRAVIS_BUILD_DIR/release-artifacts/libs/libtensorflow.so $TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Release/
cd $TRAVIS_BUILD_DIR/tests/TensorFlowSharp.Tests/bin/Release
mono xunit.console.exe TensorFlowSharp.Tests.dll xunit.console.exe.config