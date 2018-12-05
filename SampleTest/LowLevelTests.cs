﻿//
// Low-level tests
// 
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorFlow;
using System.IO;
using System.Collections.Generic;
using Learn.Mnist;
using System.Linq;

namespace SampleTest
{
	partial class MainClass
	{
		TFOperation Placeholder (TFGraph graph, TFStatus s)
		{
			var desc = new TFOperationDesc (graph, "Placeholder", "feed");
			desc.SetAttrType ("dtype", TFDataType.Int32);
			Console.WriteLine ("Handle: {0}", desc.Handle);
			var j = desc.FinishOperation ();
			Console.WriteLine ("FinishHandle: {0}", j.Handle);
			return j;
		}

		TFOperation ScalarConst (TFTensor v, TFGraph graph, TFStatus status, string name = null)
		{
			var desc = new TFOperationDesc (graph, "Const", name == null ? "scalar" : name);
			desc.SetAttr ("value", v, status);
			if (status.StatusCode != TFCode.Ok)
				return null;
			desc.SetAttrType ("dtype", TFDataType.Int32);
			return desc.FinishOperation ();
		}

		TFOperation Add (TFOperation left, TFOperation right, TFGraph graph, TFStatus status)
		{
			var op = new TFOperationDesc (graph, "AddN", "add");

			op.AddInputs (new TFOutput (left, 0), new TFOutput (right, 0));
			return op.FinishOperation ();
		}

		public void TestImportGraphDef ()
		{
			var status = new TFStatus ();
			TFBuffer graphDef;

			// Create graph with two nodes, "x" and "3"
			using (var graph = new TFGraph ()) {
				Assert (status);
				Placeholder (graph, status);
				Assert (graph ["feed"] != null);

				ScalarConst (3, graph, status);
				Assert (graph ["scalar"] != null);

				// Export to GraphDef
				graphDef = new TFBuffer ();
				graph.ToGraphDef (graphDef, status);
				Assert (status);
			};

			// Import it again, with a prefix, in a fresh graph
			using (var graph = new TFGraph ()) {
				using (var options = new TFImportGraphDefOptions ()) {
					options.SetPrefix ("imported");
					graph.Import (graphDef, options, status);
					Assert (status);
				}
				graphDef.Dispose ();

				var scalar = graph ["imported/scalar"];
				var feed = graph ["imported/feed"];
				Assert (scalar != null);

				Assert (feed != null);

				// Can add nodes to the imported graph without trouble
				Add (feed, scalar, graph, status);
				Assert (status);
			}
		}

		public void TestSession ()
		{
			var status = new TFStatus ();
			using (var graph = new TFGraph ()) {
				var feed = Placeholder (graph, status);
				var two = ScalarConst (2, graph, status);
				var add = Add (feed, two, graph, status);
				Assert (status);

				// Create a session for this graph
				using (var session = new TFSession (graph, status)) {
					Assert (status);

					// Run the graph
					var inputs = new TFOutput [] {
						new TFOutput (feed, 0)
					};
					var input_values = new TFTensor [] {
						3
					};
					var add_output = new TFOutput (add, 0);
					var outputs = new TFOutput [] {
						add_output
					};

					var results = session.Run (runOptions: null,
									   inputs: inputs,
								      inputValues: input_values,
									  outputs: outputs,
								      targetOpers: null,
								      runMetadata: null,
								     status: status);
					Assert (status);
					var res = results [0];
					Assert (res.TensorType == TFDataType.Int32);
					Assert (res.NumDims == 0); // Scalar
					Assert (res.TensorByteSize == (UIntPtr)4);
					Assert (Marshal.ReadInt32 (res.Data) == 3 + 2);

					// Use runner API
					var runner = session.GetRunner ();
					runner.AddInput (new TFOutput (feed, 0), 3);
					runner.Fetch (add_output);
					results = runner.Run (status: status);
					res = results [0];
					Assert (res.TensorType == TFDataType.Int32);
					Assert (res.NumDims == 0); // Scalar
					Assert (res.TensorByteSize == (UIntPtr)4);
					Assert (Marshal.ReadInt32 (res.Data) == 3 + 2);


				}
			}
		}

		public void TestOperationOutputListSize ()
		{
			using (var graph = new TFGraph ()) {
				var c1 = graph.Const (1L, "c1");
				var cl = graph.Const (new int [] { 1, 2 }, "cl");
				var c2 = graph.Const (new long [,] { { 1, 2 }, { 3, 4 } }, "c2");

				var outputs = graph.ShapeN (new TFOutput [] { c1, c2 });
				var op = outputs [0].Operation;

				Assert (op.OutputListLength ("output") == 2);
				Assert (op.NumOutputs == 2);
			}
		}

		public void TestOutputShape ()
		{
			using (var graph = new TFGraph ()) {
				var c1 = graph.Const (0L, "c1");
				var s1 = graph.GetShape (c1);
				var c2 = graph.Const (new long [] { 1, 2, 3 }, "c2");
				var s2 = graph.GetShape (c2);
				var c3 = graph.Const (new long [,] { { 1, 2, 3 }, { 4, 5, 6 } }, "c3");
				var s3 = graph.GetShape (c3);
			}
		}

		class WhileTester : IDisposable
		{
			public TFStatus status;
			public TFGraph graph;
			public TFSession session;
			public TFSession.Runner runner;
			public TFOutput [] inputs, outputs;

			public WhileTester ()
			{
				status = new TFStatus ();
				graph = new TFGraph ();
			}

			public void Init (int ninputs, TFGraph.WhileConstructor constructor)
			{
				inputs = new TFOutput [ninputs];
				for (int i = 0; i < ninputs; ++i)
					inputs [i] = graph.Placeholder (TFDataType.Int32, operName: "p" + i);

				Assert (status);
				outputs = graph.While (inputs, constructor, status);
				Assert (status);
			}

			public TFTensor [] Run (params int [] inputValues)
			{
				Assert (inputValues.Length == inputs.Length);

				session = new TFSession (graph);
				runner = session.GetRunner ();

				for (int i = 0; i < inputs.Length; i++)
					runner.AddInput (inputs [i], (TFTensor)inputValues [i]);
				runner.Fetch (outputs);
				return runner.Run ();
			}

			public void Dispose ()
			{
				if (session != null)
					session.Dispose ();
				if (graph != null)
					graph.Dispose ();
			}
		}

		public void WhileTest ()
		{
			using (var j = new WhileTester ()) {

				// Create loop: while (input1 < input2) input1 += input2 + 1
				j.Init (2, (TFGraph conditionGraph, TFOutput [] condInputs, out TFOutput condOutput, TFGraph bodyGraph, TFOutput [] bodyInputs, TFOutput [] bodyOutputs, out string name) => {
					Assert (bodyGraph.Handle != IntPtr.Zero);
					Assert (conditionGraph.Handle != IntPtr.Zero);

					var status = new TFStatus ();
					var lessThan = conditionGraph.Less (condInputs [0], condInputs [1]);

					Assert (status);
					condOutput = new TFOutput (lessThan.Operation, 0);

					var add1 = bodyGraph.Add (bodyInputs [0], bodyInputs [1]);
					var one = bodyGraph.Const (1);
					var add2 = bodyGraph.Add (add1, one);
					bodyOutputs [0] = new TFOutput (add2, 0);
					bodyOutputs [1] = bodyInputs [1];

					name = "Simple1";
				});

				var res = j.Run (-9, 2);

				Assert (3 == (int)res [0].GetValue ());
				Assert (2 == (int)res [1].GetValue ());
			};
		}

		// For this to work, we need to surface REGISTER_OP from C++ to C

		class AttributeTest : IDisposable
		{
			static int counter;
			public TFStatus Status;
			TFGraph graph;
			//TFOperationDesc desc;

			public AttributeTest ()
			{
				Status = new TFStatus ();
				graph = new TFGraph ();
			}

			public TFOperationDesc Init (string op)
			{
				string opname = "AttributeTest";
				if (op.StartsWith ("list(")) {
					op = op.Substring (5, op.Length - 6);
					opname += "List";
				}
				opname += op;
				return new TFOperationDesc (graph, opname, "name" + (counter++));
			}

			public void Dispose ()
			{
				graph.Dispose ();
				Status.Dispose ();
			}
		}

		void ExpectMeta (TFOperation op, string name, int expectedListSize, TFAttributeType expectedType, int expectedTotalSize)
		{
			var meta = op.GetAttributeMetadata (name);
			Assert (meta.IsList == (expectedListSize >= 0 ? true : false));
			Assert (expectedListSize == meta.ListSize);
			Assert (expectedTotalSize == meta.TotalSize);
			Assert (expectedType == meta.Type);
		}

		public void AttributesTest ()
		{
			using (var x = new AttributeTest ()) {
				var shape1 = new TFShape (new long [] { 1, 3 });
				var shape2 = new TFShape (2, 4, 6);
				var desc = x.Init ("list(shape)");
				desc.SetAttrShape ("v", new TFShape [] { shape1, shape2 });
				var op = desc.FinishOperation ();
				ExpectMeta (op, "v", 2, TFAttributeType.Shape, 5);
			}
		}

		public void GetAttributesTest ()
		{
			Console.WriteLine ("Testing attribute getting");
			var status = new TFStatus ();

			using (var graph = new TFGraph ()) {
				// Create a graph
				Assert (status);
				var desc = new TFOperationDesc (graph, "Placeholder", "node");
				desc.SetAttrType ("dtype", TFDataType.Float);
				long [] ref_shape = new long [3] { 1, 2, 3 };
				desc.SetAttrShape ("shape", new TFShape(ref_shape));
				var j = desc.FinishOperation ();
				Assert (graph ["node"] != null);

				// Check that the type is correct
				Assert (graph ["node"].GetAttributeType ("dtype", status) == TFDataType.Float);
				Assert (status);

				// Check that the shape is correct
				var metadata = graph ["node"].GetAttributeMetadata ("shape");
				Assert(Enumerable.SequenceEqual(graph["node"].GetAttributeShape ("shape", 
				                                                                 (int)metadata.TotalSize, 
				                                                                 status).ToArray(), ref_shape));
				Assert (status);

			};
		}

		public void AddControlInput ()
		{
			Console.WriteLine ("Testing AddControlInput for assertions");
			var status = new TFStatus ();
			using (var g = new TFGraph ()) {
				var s = new TFSession (g, status);

				TFTensor yes = true;
				TFTensor no = false;
				var placeholder = g.Placeholder (TFDataType.Bool, operName: "boolean");

				var check = new TFOperationDesc (g, "Assert", "assert")
					.AddInput (placeholder)
					.AddInputs (placeholder)
					.FinishOperation ();

				var noop = new TFOperationDesc (g, "NoOp", "noop")
					.AddControlInput (check)
					.FinishOperation ();

				var runner = s.GetRunner ();
				runner.AddInput (placeholder, yes);
				runner.AddTarget (noop);

				// No problems when the Assert check succeeds
				runner.Run ();

				// Exception thrown by the execution of the Assert node
				try {
					runner = s.GetRunner ();
					runner.AddInput (placeholder, no);
					runner.AddTarget (noop);
					runner.Run ();
					throw new Exception ("This should have thrown an exception");
				} catch (Exception) {
					Console.WriteLine ("Success, got the expected exception when using tensorflow control inputs to assert");
				}
			}
		}

		public void TestParametersWithIndexes ()
		{
			Console.WriteLine ("Testing Parameters with indexes");
			var status = new TFStatus ();
			using (var g = new TFGraph ()) {
				var s = new TFSession (g, status);

				var split = new TFOperationDesc (g,"Split", "Split")
					.AddInput (ScalarConst (0, g, status) [0])
					.AddInput (ScalarConst (new TFTensor (new int [] { 1, 2, 3, 4 }), g, status, "array") [0])
					.SetAttr ("num_split", 2)
					.FinishOperation ();
				var add = new TFOperationDesc (g, "Add", "Add")
					.AddInput (split [0]).AddInput (split [1]).FinishOperation () [0];

				// fetch using colon separated names
				var fetched = s.GetRunner ().Fetch ("Split:1").Run () [0];
				var vals = fetched.GetValue () as int [];
				if (vals [0] != 3 || vals [1] != 4)
					throw new Exception ("Expected the values 3 and 4");

				// Add inputs using colon separated names.
				var t = new TFTensor (new int [] { 4, 3, 2, 1 });
				var ret = (s.GetRunner ().AddInput ("Split:0", t).AddInput ("Split:1", t).Fetch ("Add").Run ()).GetValue (0) as TFTensor;
				var val = ret.GetValue () as int [];

				if (val [0] != 8 || val [1] != 6 || val [2] != 4 || val [3] != 2)
					throw new Exception ("Expected 8, 6, 4, 2");
			}
			Console.WriteLine ("success");
		}
	}
}
