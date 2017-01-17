using System;
using TensorFlow;

namespace SampleTest
{
	class MainClass
	{
		TFOperation Placeholder (TFGraph graph, TFStatus s)
		{
			var desc = new TFOperation (graph, "Placeholder", "feed");
			desc.SetAttrType ("dtype", TFDataType.Int32);
			return desc;
		}

		TFTensor Int32Tensor (int v)
		{
			// return new TFTensor (TFDataType.Int32, null, 
			return null;;
		}

		TFOperation ScalarConst (int v, TFGraph graph, TFStatus s)
		{
			var desc = new TFOperation (graph, "Const", "scalar");
			//desc.SetAttr (desc, "value"
			return null;
		}

		public void TestImportGraphDef ()
		{
			var status = new TFStatus ();
			var graph = new TFGraph ();
			Placeholder (graph, status);
		}

		public static void Main (string [] args)
		{
			Console.WriteLine (Environment.CurrentDirectory);
			Console.WriteLine ("TensorFlow version: " + TFCore.Version);
			new MainClass ().TestImportGraphDef ();
		}
	}
}
