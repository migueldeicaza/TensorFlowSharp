msbuild /t:Restore TensorFlowSharp.sln 
msbuild /p:Configuration=Release TensorFlowSharp/TensorFlowSharp.csproj 
# msbuild /t:Pack /p:Configuration=Release TensorFlowSharp/TensorFlowSharp.csproj 
unzip -l /Users/matthew/Projects/TensorFlowSharp/TensorFlowSharp/bin/Release/TensorFlowSharp.1.5.0-pre2.nupkg