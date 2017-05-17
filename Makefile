
rebuild-docs: docs/template
	mdoc export-html -o docs --template=docs/template ecmadocs/en/

# Used to fetch XML doc updates from the C# compiler into the ECMA docs
doc-update:
	mdoc update -i TensorFlowSharp/bin/Debug/TensorFlowSharp.xml -o ecmadocs/en TensorFlowSharp/bin/Debug/TensorFlowSharp.dll 
