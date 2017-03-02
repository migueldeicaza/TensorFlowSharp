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
			Assert (meta.IsList == (expectedListSize >= 0 ? true : false));
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

		#region Samples
		// 
		// Samples to exercise the API usability
		//
		// From https://github.com/aymericdamien/TensorFlow-Examples
		//
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

		//
		// Shows the use of Variable
		//
		void TestVariable ()
		{
			Console.WriteLine ("Variables");
			var status = new TFStatus ();
			using (var g = new TFGraph ()) {
				var initValue = g.Const (1.5);
				var increment = g.Const (0.5);
				TFOperation init;
				TFOutput value;
				var handle = g.Variable (initValue, out init, out value);

				// Add 0.5 and assign to the variable.
				// Perhaps using op.AssignAddVariable would be better,
				// but demonstrating with Add and Assign for now.
				var update = g.AssignVariableOp (handle, g.Add (value, increment));

				var s = new TFSession (g);
				// Must first initialize all the variables.
				s.GetRunner ().AddTarget (init).Run (status);
				Assert (status);
				// Now print the value, run the update op and repeat
				// Ignore errors.
				for (int i = 0; i < 5; i++) {
					// Read and update
					var result = s.GetRunner ().Fetch (value).AddTarget (update).Run ();

					Console.WriteLine ("Result of variable read {0} -> {1}", i, result [0].GetValue ());
				}
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

		int ArgMax (byte [,] array, int idx)
		{
			int max = -1;
			int maxIdx = -1;
			var l = array.GetLength (1);
			for (int i = 0; i < l; i++)
				if (array [idx, i] > max)
					maxIdx = i;
			return maxIdx;
		}	

		void NearestNeighbor ()
		{
			// Get the Mnist data

			var mnist = Mnist.Load ();

			// 5000 for training
			const int trainCount = 5000;
			const int testCount = 200;
			var Xtr = mnist.GetBatchReader (mnist.TrainImages).ReadAsTensor (trainCount);
			var Ytr = mnist.OneHotTrainLabels;
			var Xte = mnist.GetBatchReader (mnist.TestImages).Read (testCount);
			var Yte = mnist.OneHotTestLabels;



			Console.WriteLine ("Nearest neighbor on Mnist images");
			using (var g = new TFGraph ()) {
				var s = new TFSession (g);

				var xtr = g.Placeholder (TFDataType.Float, new TFShape (-1, 784));
				var xte = g.Placeholder (TFDataType.Float, new TFShape (784));

				// Nearest Neighbor calculation using L1 Distance
				// Calculate L1 Distance
				var distance = g.ReduceSum (g.Abs (g.Add (xtr, g.Neg (xte))), axis: g.Const (1));

				// Prediction: Get min distance index (Nearest neighbor)
				var pred = g.ArgMin (distance, g.Const (0));

				var accuracy = 0f;
				// Loop over the test data
				for (int i = 0; i < testCount; i++) {
					var runner = s.GetRunner ();

					// Get nearest neighbor

					var result = runner.Fetch (pred).AddInput (xtr, Xtr).AddInput (xte, Xte [i].DataFloat).Run ();
					var nn_index = (int)(long) result [0].GetValue ();

					// Get nearest neighbor class label and compare it to its true label
					Console.WriteLine ($"Test {i}: Prediction: {nn_index} {ArgMax (Ytr, nn_index)} True class: {ArgMax (Yte, i)}");
					if (ArgMax (Ytr, nn_index) == ArgMax (Yte, i))
						accuracy += 1f/ Xte.Length;
				}
				Console.WriteLine ("Accuracy: " + accuracy);
			}
		}

#if true
		void LinearRegression ()
		{
			Console.WriteLine ("Linear regression");
			// Parameters
			var learning_rate = 0.01;
			var training_epochs = 1000;
			var display_step = 50;

			// Training data
			var train_x = new double [] {
				3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
				7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1
			};
			var train_y = new double [] {
				1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
			 	2.827,3.465,1.65,2.904,2.42,2.94,1.3
			};
			var n_samples = train_x.Length;
			using (var g = new TFGraph ()) {
				var s = new TFSession (g);
		 		var rng = new Random ();
				// tf Graph Input

				var X = g.Placeholder (TFDataType.Float);
				var Y = g.Placeholder (TFDataType.Float);
				var W = g.Variable (g.Const (rng.Next ()), operName: "weight");
				var b = g.Variable (g.Const (rng.Next ()), operName: "bias");

				var pred = g.Add (g.Mul (X, W), b);

		// Struggling with the following:
		// The call to g.Pow returns a TFOutput, but g.ReduceSum expects a TFTensor
		// Python seems to return operation definitions, and somehow those can be p
		//passed as tensors:
		// tensorflow/python/framework/op_def_library.py
		//  (apply_op)
		//
		//https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
				var cost = g.Div (g.ReduceSum (g.Pow (g.Sub (pred, Y), g.Const (2))), g.Mul (g.Const (2), g.Const (n_samples)));

			
				// STuck here: need gradient support
			}
		}
#endif
		#endregion
		public static void Main (string [] args)
		{
			Console.WriteLine (Environment.CurrentDirectory);
			Console.WriteLine ("TensorFlow version: " + TFCore.Version);

			//var b = TFCore.GetAllOpList ();


			var t = new MainClass ();
			t.TestImportGraphDef ();
			t.TestSession ();
			t.TestOperationOutputListSize ();
			t.TestVariable ();

			// Current failing test
			t.TestOutputShape ();
			//t.AttributesTest ();
			t.WhileTest ();

			//var n = new Mnist ();
			//n.ReadDataSets ("/Users/miguel/Downloads", numClasses: 10);

			t.BasicConstantOps ();
			t.BasicVariables ();
			t.BasicMatrix ();

			t.NearestNeighbor ();

		}
	}
}
