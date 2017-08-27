using System.Collections.Generic;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class MathTests
	{
		private static IEnumerable<object []> reduceMeanData ()
		{
			// Example from https://www.tensorflow.org/api_docs/python/tf/reduce_mean
			// # 'x' is [[1., 1.]
			// #         [2., 2.]]
			//  tf.reduce_mean (x) ==> 1.5
			//  tf.reduce_mean (x, 0) ==> [1.5, 1.5]
			// 	tf.reduce_mean (x, 1) ==> [1., 2.]

			var x = new double [,] { { 1, 1 },
									 { 2, 2 } };

			yield return new object [] { x, null, 1.5 };
			yield return new object [] { x, 0, new double [] { 1.5, 1.5 } };
			yield return new object [] { x, 1, new double [] { 1, 2 } };
		}

		[Theory]
		[MemberData (nameof (reduceMeanData))]
		public void Should_ReduceMean (double [,] input, int? axis, object expected)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var tinput = graph.Placeholder (TFDataType.Double, new TFShape (2, 2));

				TFTensor [] result;
				if (axis != null) {
					var taxis = graph.Const (axis.Value);
					TFOutput y = graph.ReduceMean (tinput, taxis);
					result = session.Run (new [] { tinput, taxis }, new TFTensor [] { input, axis }, new [] { y });

					double [] actual = (double [])result [0].GetValue ();
					TestUtils.MatrixEqual (expected, actual, precision: 8);
				} else {
					TFOutput y = graph.ReduceMean (tinput, axis: null);
					result = session.Run (new [] { tinput }, new TFTensor [] { input }, new [] { y });

					double actual = (double)result [0].GetValue ();
					TestUtils.MatrixEqual (expected, actual, precision: 8);
				}
			}
		}

	}
}
