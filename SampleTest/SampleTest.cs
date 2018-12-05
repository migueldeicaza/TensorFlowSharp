﻿//
// This is just a dumping ground to exercise different capabilities 
// of the API.  Some idioms might be useful, some not, feel free to
//
// 
﻿using System;
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
				var val = results.GetValue ();
				Console.WriteLine ("a+b={0}", val);

				// Test Zeros, Ones
		                var o = g.Ones(new TFShape(4, 4));
		                var r = g.RandomNormal(new TFShape(4, 4));
		                var z = g.Zeros(new TFShape(4, 4));
		                var m = g.Mul(o, r);
		                var res1 = s.GetRunner().Run(m);
		                var res2 = s.GetRunner().Run(g.Mul(g.Mul(o, r), z));
		                
		                //Test Constants
		                var co = g.Constant(1.0, new TFShape(4, 4), TFDataType.Double);
		                var cz = g.Constant(0.0, new TFShape(4, 4), TFDataType.Double);
		                var cr = g.RandomNormal(new TFShape(4, 4));
		                var cm = g.Mul(co, cr);
		                var cres1 = s.GetRunner().Run(cm);
		                var cres2 = s.GetRunner().Run(g.Mul(g.Mul(co, cr), cz));

		                var so = g.Ones(new TFShape(4, 3));
		                var sr = g.RandomNormal(new TFShape(3, 5));
		                var sz = g.Zeros(new TFShape(5, 6));
		                var sm = g.MatMul(so, sr);
		                var sres1 = s.GetRunner().Run(sm);
		                var sres2 = s.GetRunner().Run(g.MatMul(g.MatMul(so, sr), sz));

				// Multiply two constants
				results = s.GetRunner ().Run (g.Mul (a, b));
				Console.WriteLine ("a*b={0}", results.GetValue ());

				Console.WriteLine ("graph to string: " + g.ToString ());
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
				Console.WriteLine ("a+b={0}", runner.Run (add).GetValue ());

				runner = s.GetRunner ();
				runner.AddInput (var_a, new TFTensor ((short)3));
				runner.AddInput (var_b, new TFTensor ((short)2));

				Console.WriteLine ("a*b={0}", runner.Run (mul).GetValue ());

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

		void BasicMultidimensionalArray ()
		{
			Console.WriteLine ("Basic multidimensional array");
			using (var g = new TFGraph ()) {
				var s = new TFSession (g);

				var var_a = g.Placeholder (TFDataType.Int32);
				var mul = g.Mul (var_a, g.Const (2));

				var a = new int[,,] { { { 0, 1 } , { 2, 3 } } , { { 4, 5 }, { 6, 7 } } };
				var result = s.GetRunner ().AddInput (var_a, a).Fetch (mul).Run () [0];

				var actual = (int[,,])result.GetValue ();
				var expected = new int[,,] { { { 0, 2 } , { 4, 6 } } , { { 8, 10 }, { 12, 14 } } };

				Console.WriteLine ("Actual:   " + RowOrderJoin (actual));
				Console.WriteLine ("Expected: " + RowOrderJoin (expected));
				Assert(expected.Cast<int> ().SequenceEqual (actual.Cast<int> ()));
			};
		}

		private static string RowOrderJoin(int[,,] array) => string.Join (", ", array.Cast<int> ());

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


				var result = s.GetRunner ().Run (product);
				Console.WriteLine ("Tensor ToString=" + result);
				Console.WriteLine ("Value [0,0]=" + ((double[,])result.GetValue ())[0,0]);

			};
		}

		int ArgMax (float [,] array, int idx)
		{
			float max = -1;
			int maxIdx = -1;
			var l = array.GetLength (1);
			for (int i = 0; i < l; i++)
				if (array [idx, i] > max) {
					maxIdx = i;
					max = array [idx, i];
				}
			return maxIdx;
		}

		public float [] Extract (float [,] array, int index)
		{
			var n = array.GetLength (1);
			var ret = new float [n];

			for (int i = 0; i < n; i++)
				ret [i] = array [index,i];
			return ret;
		}

		// This sample has a bug, I suspect the data loaded is incorrect, because the returned
		// values in distance is wrong, and so is the prediction computed from it.
		void NearestNeighbor ()
		{
			// Get the Mnist data

			var mnist = Mnist.Load ();

			// 5000 for training
			const int trainCount = 5000;
			const int testCount = 200;
			(var trainingImages, var trainingLabels) = mnist.GetTrainReader ().NextBatch (trainCount);
			(var testImages, var testLabels) = mnist.GetTestReader ().NextBatch (testCount);

			Console.WriteLine ("Nearest neighbor on Mnist images");
			using (var g = new TFGraph ()) {
				var s = new TFSession (g);


				TFOutput trainingInput = g.Placeholder (TFDataType.Float, new TFShape (-1, 784));

				TFOutput xte = g.Placeholder (TFDataType.Float, new TFShape (784));

				// Nearest Neighbor calculation using L1 Distance
				// Calculate L1 Distance
				TFOutput distance = g.ReduceSum (g.Abs (g.Add (trainingInput, g.Neg (xte))), axis: g.Const (1));

				// Prediction: Get min distance index (Nearest neighbor)
				TFOutput pred = g.ArgMin (distance, g.Const (0));

				var accuracy = 0f;
				// Loop over the test data
				for (int i = 0; i < testCount; i++) {
					var runner = s.GetRunner ();

					// Get nearest neighbor

					var result = runner.Fetch (pred).Fetch (distance).AddInput (trainingInput, trainingImages).AddInput (xte, Extract (testImages, i)).Run ();
					var r = result [0].GetValue ();
					var tr = result [1].GetValue ();
					var nn_index = (int)(long) result [0].GetValue ();

					// Get nearest neighbor class label and compare it to its true label
					Console.WriteLine ($"Test {i}: Prediction: {ArgMax (trainingLabels, nn_index)} True class: {ArgMax (testLabels, i)} (nn_index={nn_index})");
					if (ArgMax (trainingLabels, nn_index) == ArgMax (testLabels, i))
						accuracy += 1f/ testImages.Length;
				}
				Console.WriteLine ("Accuracy: " + accuracy);
			}
		}

#if true
		// 
		// Port of https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
		//
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

				var W = g.Variable (g.Const ((float)rng.Next ()), operName: "weight");
				var b = g.Variable (g.Const ((float) rng.Next ()), operName: "bias");
				var pred = g.Add (g.Mul (X, W.Read, "x*w"), b.Read);

				var first = g.Pow (g.Sub (pred, Y), g.Const ((float)2));
				var cost = g.Div (g.ReduceSum (g.Pow (g.Sub (pred, Y), g.Const (2f))), g.Mul (g.Const (2f), g.Const ((float)n_samples), "2*n_samples"));

				// STuck here: TensorFlow bindings need to surface gradient support
				// waiting on Google for this
				// https://github.com/migueldeicaza/TensorFlowSharp/issues/25
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
			t.LinearRegression ();
			t.TestParametersWithIndexes ();
			t.AddControlInput ();
			t.TestImportGraphDef ();
			t.TestSession ();
			t.TestOperationOutputListSize ();
			t.TestVariable ();

			// Current failing test
			t.TestOutputShape ();
			//t.AttributesTest ();
			t.GetAttributesTest ();
			t.WhileTest ();

			//var n = new Mnist ();
			//n.ReadDataSets ("/Users/miguel/Downloads", numClasses: 10);

			t.BasicConstantOps ();
			t.BasicVariables ();
			t.BasicMultidimensionalArray ();
			t.BasicMatrix ();

			t.NearestNeighbor ();

		}
	}
}
