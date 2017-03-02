﻿//
// This is the driver for the operation generator, this takes data that
// is provided by the Tensorflow runtime to produce strongly-typed and
// high level methods on the TFGraph class.
//
// The result is generated into a partial class that is lined with the
// main TensorFlowSharp library
//
// Authors:
//   Miguel de Icaza
//
// Copyright 2017, the year of downfall, Microsoft Inc
//
#pragma warning disable RECS0063 // Warns when a culture-aware 'StartsWith' call is used by default.

using System;
using System.Collections.Generic;
using System.IO;
using ProtoBuf;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using tensorflow;

class OpGenerator
{
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
			cstype = "TFShape"; break;
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

	bool IsReferenceType (string tfType)
	{
		if (tfType.StartsWith ("list("))
			return true;
		if (tfType == "tensor" || tfType == "string" || tfType == "shape")
			return true;
		return false;
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
		case "ref":
			return "reference";
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
	bool return_is_tfoutput;

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
		// API: currently, if we have a single ref TFOutput result, we make the signature of the 
		// function return that TFOutput instead of the TFOperation (as you can get the TFOperation
		// from the TFOutput anyways.
		//
		// When we move to tuples, we could probably put everything in a Tuple result, but for now
		// mult-return functions will just return all outputs on ref variables, instead of the first
		// as a ref, and the rest as TFOutputs.
		//
		// This means that we generate methods like this:
		//    TFOutput Constant (....)
		// when there is a single output
		//
		//    TFOperation Foo (..)
		// When there is no result or more than one result.
		return_is_tfoutput = def.output_arg.Count == 1;

	}

	// Generates arguments:
	//   * Input arguments (TFOutput or TFOutput [])
	//   * All required attributes
	//   * variadic optional arguments
	string FillArguments (OpDef def)
	{
		var sb = new StringBuilder ();
		string comma = "";
		foreach (var inarg in def.input_arg) {
			string type = "TFOutput" + (IsListArg (inarg) ? "[]" : "");

			sb.AppendFormat ($"{comma}{type} {ParamMap (inarg.name)}");
			comma = ", ";
		}
		foreach (var attr in required_attrs) {
			sb.AppendFormat ($"{comma}{CSharpType (attr.type)} {ParamMap (attr.name)}");
			comma = ", ";
		}

		if (!return_is_tfoutput) {
			foreach (var arg in def.output_arg) {
				string type = "TFOutput" + (IsListArg (arg) ? "[]" : "");

				sb.AppendFormat ($"{comma}ref {type} {ParamMap (arg.name)}");
				comma = ", ";
			}
		}

		int n = 0;
		foreach (var attr in optional_attrs) {
			bool reftype = IsReferenceType (attr.type);
			var cstype = CSharpType (attr.type);
			var cstypesuffix = reftype ? "" : "?";

			sb.AppendFormat ($"{comma}{cstype}{cstypesuffix} {attr.name} = null");
			comma = ", ";
		}
		if (sb.Length != 0)
			sb.Append (", ");
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
		if (!return_is_tfoutput) {
			foreach (var attr in oper.output_arg) {
				if (String.IsNullOrEmpty (attr.description))
					continue;
				p ($"/// <param name=\"{ParamMap (attr.name)}\">");
				Comment (attr.description);
				p ($"/// </param>");
			}
		}
		p ("/// <param name=\"operName\">");
		p ($"///   If specified, the created operation in the graph will be this one, otherwise it will be named '{oper.name}'.");
		p ("/// </param>");
		foreach (var attr in optional_attrs) {
			if (String.IsNullOrEmpty (attr.description))
				continue;
			p ($"/// <param name=\"{ParamMap (attr.name)}\">");
			Comment ("Optional argument");
			Comment (attr.description);
			p ($"/// </param>");
		}

		if (return_is_tfoutput) {
			p ($"/// <returns>");
			Comment (oper.output_arg.First ().description);
			p ($"/// </returns>");
		}
		if (!String.IsNullOrEmpty (oper.description)) {
			p ("/// <remarks>");
			Comment (oper.description);
			p ("/// </remarks>");
		}
	}

	void SetAttribute (string type, string attrName, string csAttrName)
	{
		if (type == "shape") {
			p ($"desc.SetAttrShape (\"{attrName}\", {csAttrName});");
			return;
		}
		if (type.StartsWith ("list(shape")) {
			p ($"desc.SetAttrShape (\"{attrName}\", {csAttrName});");
			return;
		}

		var cstype = CSharpType (type);
		switch (cstype) {
		case "long":
		case "long[]":
		case "string":
		case "string[]":
		case "float":
		case "float[]":
		case "bool":
		case "bool[]":
			p ($"desc.SetAttr (\"{attrName}\", {csAttrName});");
			break;
		case "TFDataType":
		case "TFDataType[]":
			p ($"desc.SetAttrType (\"{attrName}\", {csAttrName});");
			break;

		// This should pass the cstatus, but requires the 
		// function to take a TFStatus as well, so need to weave that
		// in the parameters
		case "TFTensor":
		case "TFTensor[]":
			p ($"desc.SetAttr (\"{attrName}\", {csAttrName} /* cstatus */);");
			break;
		default:
			throw new Exception ("Unexpected type: " + cstype);
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
		string retType;

		if (return_is_tfoutput) {
			if (oper.output_arg.Any (x => IsListArg (x)))
				retType = "TFOutput []";
			else
				retType = "TFOutput";
		} else
			retType = "TFOperation";
		

		p ($"public {retType} {name} ({FillArguments(oper)}string operName = null)");
		pi ("{");
		bool needStatus = required_attrs.Concat (optional_attrs).Any (attr => attr.type.Contains ("TFTensor"));
		p ($"var desc = new TFOperationDesc (this, \"{oper.name}\", MakeName (\"{oper.name}\", operName));");
		foreach (var arg in oper.input_arg) {
			if (IsListArg (arg))
				p ($"desc.AddInputs ({ParamMap (arg.name)});");
			   else
				p ($"desc.AddInput ({ParamMap (arg.name)});");
		}

		// If we have attributes
		if (required_attrs.Count > 0 || optional_attrs.Count > 0) {
			foreach (var attr in required_attrs) {
				SetAttribute (attr.type, attr.name, ParamMap (attr.name));
			}

			foreach (var attr in optional_attrs) {
				var reftype = IsReferenceType (attr.type);
				var csattr = ParamMap (attr.name);
				if (reftype)
					pi ($"if ({csattr} != null)");
				else
					pi ($"if ({csattr}.HasValue)");
				SetAttribute (attr.type, attr.name, csattr + (reftype ? "" : ".Value"));
				pd ("");

			}
		}

		p ("var op = desc.FinishOperation ();");
		if (oper.output_arg.Any (x => IsListArg (x))) {
			p ("int _idx = 0, _n = 0;");
			foreach (var arg in oper.output_arg) {
				string retDecl = "", retOutput;

				if (return_is_tfoutput){
					retDecl = "var ";
					retOutput = "_ret";
				} else
					retOutput = ParamMap (arg.name);

				if (IsListArg (arg)) {
					var outputs = new StringBuilder ();
					p ($"_n = op.OutputListLength (\"{arg.name}\");");
					p ($"{retDecl}{retOutput} = new TFOutput [_n];");
					pi ("for (int i = 0; i < _n; i++)");
					p ($"{retOutput} [i] = new TFOutput (op, _idx++);");
					pd ("");
					if (return_is_tfoutput)
						p ($"return {retOutput};");
				} else {
					if (return_is_tfoutput) {
						p ($"return  new TFOutput (op, _idx++);");
					} else {
						p ($"{retOutput} = new TFOutput (op, _idx++);");
					}
				}
			}
		} else {
			int idx = 0;
			foreach (var arg in oper.output_arg) {
				if (return_is_tfoutput)
					p ($"return new TFOutput (op, 0);");
				else
					p ($"{ParamMap (arg.name)} = new TFOutput (op, {idx++});");
			}
		}
		if (!return_is_tfoutput)
			p ("return op;");
		pd ("}\n");
	}

	[StructLayout (LayoutKind.Sequential)]
	internal struct LLBuffer
	{
		internal IntPtr data;
		internal IntPtr length;
		internal IntPtr data_deallocator;
	}

	[DllImport ("libtensorflow")]
	unsafe extern static LLBuffer *TF_GetAllOpList ();

	MemoryStream GetOpsList ()
	{
		unsafe
		{
			LLBuffer* ptr = TF_GetAllOpList ();
			var ret = new byte [(int)ptr->length];
			Marshal.Copy (ptr->data, ret, 0, (int)ptr->length);
			return new MemoryStream (ret);
		}
	}

	void Run ()
	{
		
		output = File.CreateText ("../../../TensorFlowSharp/Operations.g.cs");
	     	var operations = Serializer.Deserialize<List<OpDef>> (GetOpsList ());
		p ("using System;\n");

		pi ("namespace TensorFlow {");
		pi ("public partial class TFGraph {");
		foreach (var oper in operations){
			// Skip internal operations
			if (oper.name.StartsWith ("_"))
				continue;

			// Ignore functions where we lack a C# type mapping
			if (oper.attr.Any (attr => CSharpType (attr.type) == null)) {
				var attr = oper.attr.First (a => CSharpType (a.type) == null);

				Console.WriteLine ($"SkipTYPE: {oper.name} due to attribute ({attr.type} {attr.name}) lacking a mapping to C#");
				continue;
			}

			#if true
			// Ignore reference types as well (per go's binding)
			if (oper.input_arg.Any (ia => ia.is_ref)) {
				var pars = String.Join (", ", oper.input_arg.Where (x => x.is_ref).Select (x => $"{x.type} {x.name}"));
				Console.WriteLine ($"SkipInREF: {oper.name} parameters with is_ref: {pars}");
				continue;
			}

			// Ignore reference types as well (per go's binding)
			if (oper.output_arg.Any (ia => ia.is_ref)) {
				var pars = String.Join (", ", oper.input_arg.Where (x => x.is_ref).Select (x => $"{x.type} {x.name}"));
				Console.WriteLine ($"SkipOutREF: {oper.name} parameters with is_ref: {pars}");

				continue;
			}
			#endif

			// Undocumented operation, perhaps we should not surface
			if (oper.summary == "")
				continue;

			Generate (oper);
		}
		pd ("}");
		pd ("}");
		output.Close ();
	}

	// The output file
	StreamWriter output;

	int indent = 0;

	// Convenience methods to generate output
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
