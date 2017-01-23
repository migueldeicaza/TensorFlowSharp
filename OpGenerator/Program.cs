using System;
using System.Collections.Generic;
using System.IO;
using ProtoBuf;
using TensorFlow;
using tensorflow;
using System.Linq;
using System.Text;

class OpGenerator
{
	StreamWriter output;

	//
	// Maps a TensorFlow type to a C# type
	//
	string CSharpType (string tfType)
	{
		bool list = false;
		string cstype;

		if (tfType.StartsWith ("list(")) {
			list = true;
			tfType = tfType.Substring (5, tfType.Length - 6);
		}
		switch (tfType) {
		case "int":
			cstype = "long"; break;
		case "float":
			cstype = "float"; break;
		case "bool":
			cstype = "bool"; break;
		case "type":
			cstype = "TFDataType"; break;
		case "shape":
			cstype = "long[]"; break;
		case "tensor":
			cstype = "TFTensor"; break;
		case "string":
			cstype = "string"; break;
		default:
			Console.WriteLine ("Unknown data TensorFlow type: {0}", tfType);
			return null;
		}

		return cstype + (list ? "[]" : "");
	}

	// Maps a parameter name to a C# acceptable name, to avoid clashes with 
	// language keywords
	string ParamMap (string paramName)
	{
		switch (paramName) {
		case "out":
			return "output";
		case "params":
			return "parameters";
		}
		return paramName;
	}

	// Determines if the specified ArgDef represents a TensorFlow list
	bool IsListArg (OpDef.ArgDef arg)
	{
		return arg.type_list_attr != "" || arg.number_attr != "";
	}

	// 
	// These values are the result of calling SetupArguments
	//
	Dictionary<string, bool> inferred_input_args;
	List<OpDef.AttrDef> required_attrs, optional_attrs;

	void SetupArguments (OpDef def)
	{
		// Attributes related to the InputArg's type are inferred automatically
		// and are not exposed to the client.
		var inferred_input_args = new Dictionary<string, bool> ();
		required_attrs = new List<OpDef.AttrDef> ();
		optional_attrs = new List<OpDef.AttrDef> ();

		foreach (var argdef in def.input_arg) {
			if (argdef.type_attr != "")
				inferred_input_args [argdef.type_attr] = true;
			else if (argdef.type_list_attr != "")
				inferred_input_args [argdef.type_list_attr] = true;
			if (argdef.number_attr != "")
				inferred_input_args [argdef.number_attr] = true;
		}
		foreach (var attr in def.attr) {
			if (inferred_input_args.ContainsKey (attr.name))
				continue;
			if (attr.default_value == null)
				required_attrs.Add (attr);
			else
				optional_attrs.Add (attr);
		}
	}

	// Generates arguments:
	//   * Input arguments (TFOutput or TFOutput [])
	//   * All required attributes
	//   * variadic optional arguments
	string FillArguments (OpDef def)
	{
		var sb = new StringBuilder ();
		foreach (var inarg in def.input_arg) {
			string type = "TFOutput" + (IsListArg (inarg) ? "[]" : "");

			sb.AppendFormat ($", {type} {ParamMap (inarg.name)}");
		}
		foreach (var attr in required_attrs) 
			sb.AppendFormat ($", {CSharpType (attr.type)} {ParamMap (attr.name)}");

		foreach (var arg in def.output_arg) {
			string type = "TFOutput" + (IsListArg (arg) ? "[]" : "");

			sb.AppendFormat ($", ref {type} {ParamMap (arg.name)}");
		}

		// FIXME: finish this part
		int n = 0;
		foreach (var attr in optional_attrs)
			sb.AppendFormat ($", object optional{n++}");
		return sb.ToString ();
	}

	void Comment (string text)
	{
		if (text == null || text == "")
			return;
		var lines = text.Split ('\n');
		foreach (var line in lines) {
			p ($"///   {line}");
		}
	}

	// Produces the C# inline documentation
	void GenDocs (OpDef oper)
	{
		p ("/// <summary>");
		Comment (oper.summary);
		p ("/// </summary>");
		foreach (var input in oper.input_arg) {
			if (String.IsNullOrEmpty (input.description))
				continue;
			p ($"/// <param name=\"{ParamMap(input.name)}\">");
			Comment (input.description);
			p ($"/// </param>");
		}
		foreach (var attr in oper.output_arg) {
			if (String.IsNullOrEmpty (attr.description))
				continue;
			p ($"/// <param name=\"{ParamMap (attr.name)}\">");
			Comment (attr.description);
			p ($"/// </param>");
		}
		p ("/// <param name=\"operName\">");
		p ($"///   If specified, the created operation in the graph will be this one, otherwise it will be named '{oper.name}'.");
		p ("/// </param>");
		if (!String.IsNullOrEmpty (oper.description)) {
			p ("/// <remarks>");
			Comment (oper.description);
			p ("/// </remarks>");
		}
	}

	/// <summary>
	/// Generate the specified oper.
	/// </summary>
	/// <param name="oper">Oper.</param>
	void Generate (OpDef oper)
	{
		SetupArguments (oper);
		GenDocs (oper);

		var name = oper.name;
		
		p ($"public TFOperation {name} (Scope scope{FillArguments(oper)}, string operName = null)");
		pi ("{");
		bool needStatus = required_attrs.Concat (optional_attrs).Any (attr => attr.type.Contains ("TFTensor"));
		p ($"var desc = new TFOperationDesc (this, operName, operName == null ? \"{oper.name}\" : operName);");
		foreach (var arg in oper.input_arg) {
			if (IsListArg (arg))
				p ($"desc.AddInputs ({ParamMap (arg.name)});");
			   else
				p ($"desc.AddInput ({ParamMap (arg.name)});");
		}

		// If we have attributes
		if (required_attrs.Count > 0 || optional_attrs.Count > 0) {
			foreach (var attr in required_attrs) {
				var cstype = CSharpType (attr.type);
				switch (cstype) {
				case "int":
				case "int[]":
				case "string":
				case "string[]":
				case "float":
				case "float[]":
				case "bool":
				case "bool[]":
					p ($"desc.SetAttr (\"{attr.name}\", {ParamMap(attr.name)});");
					break;
				case "TFDataType":
				case "TFDataType[]":
					p ($"desc.SetAttrType (\"{attr.name}\", {ParamMap (attr.name)});");
					break;

					// This should pass the cstatus, but requires the 
					// function to take a TFStatus as well, so need to weave that
					// in the parameters
				case "TFTensor":
				case "TFTensor[]":
					p ($"desc.SetAttr (\"{attr.name}\", {ParamMap (attr.name)} /* cstatus */);");
					break;
				}
			}
		}
		p ("var op = desc.FinishOperation ();");
		if (oper.output_arg.Any (x => IsListArg (x))) {
			p ("int _idx = 0, _n = 0;");
			foreach (var arg in oper.output_arg) {
				
				if (IsListArg (arg)) {
					var outputs = new StringBuilder ();
					p ("_n = op.InputListLength (\"arg.name\");");
					p ($"{ParamMap (arg.name)} = new TFOutput [_n];");
					pi ("for (int i = 0; i < _n; i++)");
					p ($"{ParamMap (arg.name)} [i] = new TFOutput (op, _idx++);");
					pd ("");
				} else
					p ($"{ParamMap (arg.name)} = new TFOutput (op, _idx++);");
			}
		} else {
			int idx = 0;
			foreach (var arg in oper.output_arg) {
				p ($"{ParamMap (arg.name)} = new TFOutput (op, {idx++});");
			}
		}
		p ("return op;");
		pd ("}\n");
	}

	void Run ()
	{
		
		output = File.CreateText ("../../../TensorFlowSharp/Operations.cs");

		var operations = Serializer.Deserialize<List<OpDef>> (new MemoryStream (TFCore.GetAllOpList ().ToArray ()));
		pi ("namespace TensorFlow {");
		pi ("public partial class TFGraph {");
		foreach (var oper in operations){
			// Skip internal operations
			if (oper.name.StartsWith ("_"))
				continue;

			// Ignore functions where we lack a C# type mapping
			if (oper.attr.Any (attr => CSharpType (attr.type) == null))
				continue;

			// Ignore reference types as well (per go's binding)
			if (oper.input_arg.Any (ia => ia.is_ref))
				continue;
			
			// Ignore reference types as well (per go's binding)
			if (oper.output_arg.Any (ia => ia.is_ref))
				continue;

			// Undocumented operation, perhaps we should not surface
			if (oper.summary == "")
				continue;

			Generate (oper);
		}
		pd ("}");
		pd ("}");
		output.Close ();
	}

	int indent = 0;

	void pi (string fmt, params object [] args)
	{
		p (fmt, args);
		indent++;
	}

	void pd (string fmt, params object [] args)
	{
		indent--;
		p (fmt, args);
	}

	void p (string fmt, params object [] args)
	{
		for (int i = 0; i < indent; i++)
		     output.Write ("\t");
		if (args.Length == 0)
			output.WriteLine (fmt);
		else
			output.WriteLine (fmt, args);
	}

	public static void Main (string [] args)
	{
		new OpGenerator ().Run ();
	}
}
