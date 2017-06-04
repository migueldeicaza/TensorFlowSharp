all: doc-update yaml

rebuild-docs: docs/template
	mdoc export-html --force-update -o docs --template=docs/template ecmadocs/en/

# Used to fetch XML doc updates from the C# compiler into the ECMA docs
doc-update:
	mdoc update -i TensorFlowSharp/bin/Debug/TensorFlowSharp.xml -o ecmadocs/en TensorFlowSharp/bin/Debug/TensorFlowSharp.dll 

yaml:
	-rm ecmadocs/en/ns-.xml
	mono /cvs/ECMA2Yaml/ECMA2Yaml/ECMA2Yaml/bin/Debug/ECMA2Yaml.exe --source=`pwd`/ecmadocs/en --output=`pwd`/docfx/api
	(cd docfx; mono ~/Downloads/docfx/docfx.exe build)

