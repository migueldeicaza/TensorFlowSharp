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

	Dictionary<string, bool> inferred_input_args;
	List<OpDef.AttrDef> required_attrs, optional_attrs;

	// Generates arguments:
	//   * Input arguments (TFOutput or TFOutput [])
	//   * All required attributes
	//   * variadic optional arguments
	string FillArguments (OpDef def)
	{
		// Attributes related to the InputArg's type are inferred automatically
		// and are not exposed to the client.
		var inferred_input_args = new Dictionary<string, bool> ();
		var required_attrs = new List<OpDef.AttrDef> ();
		var optional_attrs = new List<OpDef.AttrDef> ();

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
		         
		var sb = new StringBuilder ();
		foreach (var inarg in def.input_arg) {
			bool isList = inarg.type_attr != "" || inarg.number_attr != "";
			string type = "TFOutput" + (isList ? "[]" : "");

			sb.AppendFormat ($", {type} {inarg.name}");
		}
		foreach (var attr in required_attrs) 
			sb.AppendFormat ($", {CSharpType (attr.type)} {attr.name}");

		// FIXME: finish this part
		foreach (var attr in optional_attrs)
			sb.AppendFormat ($", object optional");
		return sb.ToString ();
	}

	void Generate (OpDef oper)
	{
		
		var name = oper.name;
		p ($"public TFOperation {name} (Scope scope {FillArguments(oper)})");
		pi ("{");
		pd ("}\n");
	}

	void Run ()
	{
		output = File.CreateText ("/tmp/Operations.cs");

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
