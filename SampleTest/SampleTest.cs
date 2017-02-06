using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorFlow;
using System.IO;
using System.Collections.Generic;
using Learn.Mnist;

namespace SampleTest
{
	class MainClass
	{
		static public void Assert (bool assert, [CallerMemberName] string caller = null, string message = "")
		{
			if (!assert){
				throw new Exception ($"{caller}: {message}");
			}
		}

		static public void Assert (TFStatus status, [CallerMemberName] string caller = null, string message = "")
		{
			if (status.StatusCode != TFCode.Ok) {
				throw new Exception ($"{caller}: {status.StatusMessage} {message}");
			}
		}

		TFOperation Placeholder (TFGraph graph, TFStatus s)
		{
			var desc = new TFOperationDesc (graph, "Placeholder", "feed");
			desc.SetAttrType ("dtype", TFDataType.Int32);
			Console.WriteLine ("Handle: {0}", desc.Handle);
			var j = desc.FinishOperation ();
			Console.WriteLine ("FinishHandle: {0}", j.Handle);
			return j;
		}

		TFOperation ScalarConst (int v, TFGraph graph, TFStatus status)
		{
			var desc = new TFOperationDesc (graph, "Const", "scalar");
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

					var results = session.Run (    runOptions: null,
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
					Assert (res.TensorByteSize == (UIntPtr) 4);
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
				var cl = graph.Const (new int []{ 1, 2 }, "cl");
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

		// For this to work, we need to surface REGISTER_OP from C++ to C

		class AttributeTest : IDisposable
		{
			static int counter;
			public TFStatus Status;
			TFGraph graph;
			TFOperationDesc desc;

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
			Assert (meta.IsList == (expectedListSize >= 0 ? 1 : 0));
			Assert (expectedListSize == meta.ListSize);
			Assert (expectedTotalSize == expectedTotalSize);
			Assert (expectedType == meta.Type);
		}

		public void AttributesTest ()
		{
			using (var x = new AttributeTest ()) {
				var shape1 = new TFShape (new long [] { 1, 3 });
				var shape2 = new TFShape ( 2, 4, 6 );
				var desc = x.Init ("list(shape)");
				desc.SetAttrShape ("v", new TFShape [] { shape1, shape2 });
				var op = desc.FinishOperation ();
				ExpectMeta (op, "v", 2, TFAttributeType.Shape, 5);
			}
					
		}


		public static void p (string p)
		{
			Console.WriteLine (p);
		}

		void BasicConstantOps ()
		{
			//
			// Test the manual GetRunner, this could be simpler
			// we should at some point allow Run (a+b);
			//
			// The session implicitly creates the graph, get it.
			using (var s = new TFSession ()){
				var g = s.Graph;

				var a = g.Const (2);
				var b = g.Const (3);
				Console.WriteLine ("a=2 b=3");

				// Add two constants
				var results = s.GetRunner ().Run (g.Add (a, b));
				var val = results [0].GetValue ();
				Console.WriteLine ("a+b={0}", val);

				// Multiply two constants
				results = s.GetRunner ().Run (g.Mul (a, b));
				Console.WriteLine ("a*b={0}", results [0].GetValue ());

				// TODO: API-wise, perhaps session.Run () can have a simple
				// overload where we only care about the fetched values, 
				// making the above:
				// s.Run (g.Mul (a, b));
			}
		}

		// 
		// Shows how to use placeholders to pass values
		//
		void BasicVariables ()
		{
			Console.WriteLine ("Using placerholders");
			using (var g = new TFGraph ()) {
				var s = new TFSession (g);

				// We use "shorts" here, so notice the casting to short to get the
				// tensor with the right data type.
				var var_a = g.Placeholder (TFDataType.Int16);
				var var_b = g.Placeholder (TFDataType.Int16);

				var add = g.Add (var_a, var_b);
				var mul = g.Mul (var_a, var_b);

				var runner = s.GetRunner ();
				runner.AddInput (var_a, new TFTensor ((short)3));
				runner.AddInput (var_b, new TFTensor ((short)2));
				Console.WriteLine ("a+b={0}", runner.Run (add) [0].GetValue ());

				runner = s.GetRunner ();
				runner.AddInput (var_a, new TFTensor ((short)3));
				runner.AddInput (var_b, new TFTensor ((short)2));

				Console.WriteLine ("a*b={0}", runner.Run (mul) [0].GetValue ());

				// TODO
				// Would be nice to have an API that allows me to pass the values at Run time, easily:
				// s.Run (add, { var_a: 3, var_b: 2 })
				// C# allows something with Dictionary constructors, but you still must provide the type
				// signature.
			}
		}

		void BasicMatrix ()
		{
			Console.WriteLine ("Basic matrix");
			using (var g = new TFGraph ()) {
				var s = new TFSession (g);

				// 1x2 matrix
				var matrix1 = g.Const (new double [,] { { 3, 3 } });
				// 2x1 matrix
				var matrix2 = g.Const (new double [,] { { 2 }, { 2 } });

				// multiply
				var product = g.MatMul (matrix1, matrix2);


				var result = s.GetRunner ().Run (product) [0];
				Console.WriteLine ("Tensor ToString=" + result);
				Console.WriteLine ("Value [0,0]=" + ((double[,])result.GetValue ())[0,0]);

			};
		}

		public static void Main (string [] args)
		{
			Console.WriteLine (Environment.CurrentDirectory);
			Console.WriteLine ("TensorFlow version: " + TFCore.Version);

			//var b = TFCore.GetAllOpList ();


			var t = new MainClass ();
			t.TestImportGraphDef ();
			t.TestSession ();
			t.TestOperationOutputListSize ();

			// Current failing test
			t.TestOutputShape ();
			//t.AttributesTest ();


			//var n = new Mnist ();
			//n.ReadDataSets ("/Users/miguel/Downloads", numClasses: 10);

			t.BasicConstantOps ();
			t.BasicVariables ();
			t.BasicMatrix ();
		}
	}
}
