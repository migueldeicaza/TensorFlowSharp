using System;
using System.Runtime.CompilerServices;
using TensorFlow;

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
			var desc = new TFOperation (graph, "Placeholder", "feed");
			desc.SetAttrType ("dtype", TFDataType.Int32);
			Console.WriteLine ("Handle: {0}", desc.Handle);
			var j = desc.FinishOperation ();
			Console.WriteLine ("FinishHandle: {0}", j.Handle);
			return j;
		}

		TFOperation ScalarConst (int v, TFGraph graph, TFStatus status)
		{
			var desc = new TFOperation (graph, "Const", "scalar");
			desc.SetAttr ("value", TFTensor.Constant (v), status);
			if (status.StatusCode != TFCode.Ok)
				return null;
			desc.SetAttrType ("dtype", TFDataType.Int32);
			return desc.FinishOperation ();
		}

		TFOperation Add (TFOperation left, TFOperation right, TFGraph graph, TFStatus status)
		{
			var op = new TFOperation (graph, "AddN", "add");

			op.AddInputs (new TFOutput (left, 0), new TFOutput (right, 0));
			op.FinishOperation ();
			return op;
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
					graph.ImportGraphDef (graphDef, options, status);
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

				using (var session = new TFSession (graph, status)) {
					
				}
			}
		}

		public static void Main (string [] args)
		{
			Console.WriteLine (Environment.CurrentDirectory);
			Console.WriteLine ("TensorFlow version: " + TFCore.Version);
			var t = new MainClass ();
			t.TestImportGraphDef ();
			t.TestSession ();
		}
	}
}
