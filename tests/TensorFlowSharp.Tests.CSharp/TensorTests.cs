using System;
using System.Collections.Generic;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class TensorTests
	{
		private static IEnumerable<object []> jaggedData ()
		{
			yield return new object [] {
				new double [][] { new [] { 1.0, 2.0 }, new [] { 3.0, 4.0 } },
				new double [,] { { 1.0, 2.0}, { 3.0, 4.0 } },
				true
			};

			yield return new object [] {
				new double [][] { new [] { 1.0, 2.0 }, new [] { 1.0, 4.0 } },
				new double [,] { { 1.0, 2.0}, { 3.0, 4.0 } },
				false
			};

			yield return new object [] {
				new double [][][] { new [] { new [] { 1.0 }, new[] { 2.0 } }, new [] { new [] { 3.0 }, new [] { 4.0 } } },
				new double [,,] { { { 1.0 }, { 2.0 } }, { { 3.0 }, { 4.0 } } },
				true
			};

			yield return new object [] {
				new double [][][] { new [] { new [] { 1.0 }, new[] { 2.0 } }, new [] { new [] { 1.0 }, new [] { 4.0 } } },
				new double [,,] { { { 1.0 }, { 2.0 } }, { { 3.0 }, { 4.0 } } },
				false
			};
		}


		[Theory]
		[MemberData (nameof (jaggedData))]
		public void Should_MultidimensionalAndJaggedBeEqual (Array jagged, Array multidimensional, bool expected)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var tjagged = graph.Const (new TFTensor (jagged));
				var tmultidimensional = graph.Const (new TFTensor (multidimensional));

				TFOutput y = graph.Equal (tjagged, tmultidimensional);
				TFOutput r;
				if (multidimensional.Rank == 2)
					r = graph.All (y, graph.Const (new [] { 0, 1 }));
				else if (multidimensional.Rank == 3)
					r = graph.All (y, graph.Const (new [] { 0, 1, 2 }));
				else
					throw new System.Exception ("If you want to test Ranks > 3 please handle this extra case manually.");

				TFTensor [] result = session.Run (new TFOutput [] { }, new TFTensor [] { }, new [] { r });

				bool actual = (bool)result [0].GetValue ();
				Assert.Equal (expected, actual);
			}
		}

	}
}
