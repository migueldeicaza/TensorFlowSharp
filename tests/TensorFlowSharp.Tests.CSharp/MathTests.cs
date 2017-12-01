using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class MathTests
	{
		[Fact]
		public void Should_CalculateTanhGrad_Correctly ()
		{
			using (TFGraph graph = new TFGraph ())
			using (TFSession session = new TFSession (graph)) 
			{

				TFOutput x = graph.Const (new TFTensor (0.7));
				TFOutput y = graph.Tanh (x);
				TFOutput dy = graph.Const (new TFTensor (new [] { 1.0 }));
				TFOutput grad = graph.TanhGrad (y, dy);

				TFTensor [] result = session.Run (new TFOutput [] { }, new TFTensor [] { }, new [] { grad });

				double value = (double)result [0].GetValue ();
				Assert.Equal (0.634739589982459, value, 15);
			}
		}

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

		private static IEnumerable<object []> sigmoidCrossEntropyData ()
		{
			yield return new object [] { new [] { 1.0, 0.0, 1.0, 1.0 }, new [] { 1.0, 0.0, 1.0, 1.0 }, new [] { 0.31326168751822281, 0.69314718055994529, 0.31326168751822281, 0.31326168751822281 } };
			yield return new object [] { new [] { 1.0, 0.0, 1.0, 1.0 }, new [] { -0.2, 4.2, 0.0, 0.0 }, new [] { 0.79813886938159184, 4.2148842546719187, 0.69314718055994529, 0.69314718055994529 } };
			yield return new object [] { new [] { 1.0, 0.0 }, new [] { -2.1, -2, -4, 3.0 }, null };
		}

		[Theory]
		[MemberData (nameof (sigmoidCrossEntropyData))]
		public void Should_SigmoidCrossEntropyWithLogits (double [] labels, double [] logits, double [] expected) 
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var tlabels = graph.Placeholder (TFDataType.Double, new TFShape (2, 2));
				var tlogits = graph.Placeholder (TFDataType.Double, new TFShape (2, 2));

				TFOutput y = graph.SigmoidCrossEntropyWithLogits (tlabels, tlogits);

				if (expected != null) {
					TFTensor [] result = session.Run (new [] { tlabels, tlogits }, new TFTensor [] { labels, logits }, new [] { y });

					double [] actual = (double [])result [0].GetValue ();
					TestUtils.MatrixEqual (expected, actual, precision: 8);
				} else {
					Assert.Throws<TFException> (() => session.Run (new [] { tlabels, tlogits }, new TFTensor [] { labels, logits }, new [] { y }));
				}
			}
		}

		private static IEnumerable<object []> reduceProdData ()
		{
			// Example from https://www.tensorflow.org/api_docs/python/tf/reduce_mean but adapted to return prod
			var x = new double [,] { { 1, 1 },
									 { 2, 2 } };

			yield return new object [] { x, null, 4.0 };
			yield return new object [] { x, 0, new double [] { 2, 2 } };
			yield return new object [] { x, 1, new double [] { 1, 4 } };
		}

		[Theory]
		[MemberData (nameof (reduceProdData))]
		public void Should_ReduceProd (double [,] input, int? axis, object expected)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var tinput = graph.Placeholder (TFDataType.Double, new TFShape (2, 2));

				TFTensor [] result;
				if (axis != null) {
					var taxis = graph.Const (axis.Value);
					TFOutput y = graph.ReduceProd (tinput, taxis);
					result = session.Run (new [] { tinput, taxis }, new TFTensor [] { input, axis }, new [] { y });

					double [] actual = (double [])result [0].GetValue ();
					TestUtils.MatrixEqual (expected, actual, precision: 8);
				} else {
					TFOutput y = graph.ReduceProd (tinput, axis: null);
					result = session.Run (new [] { tinput }, new TFTensor [] { input }, new [] { y });

					double actual = (double)result [0].GetValue ();
					TestUtils.MatrixEqual (expected, actual, precision: 8);
				}
			}
		}

		private static IEnumerable<object []> reduceProdData2 ()
		{
			yield return new object [] { null, 170170.0 };
			yield return new object [] { -3, new [] { 1.0 } };
			yield return new object [] { -2, new [] { 22.0, 65.0, 119.0 } };
			yield return new object [] { -1, new [] { 70.0, 2431.0 } };
			yield return new object [] { 0, new [] { 22.0, 65.0, 119.0 } };
			yield return new object [] { 1, new [] { 70.0, 2431.0 } };
			yield return new object [] { 2, new [] { 1.0 } };
			yield return new object [] { 3, new [] { 1.0 } };
		}

		[Theory]
		[MemberData (nameof (reduceProdData2))]
		public void Should_ReduceProd2 (int? axis, object expected)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {

				double [,] test = {
					{ 2,   5,  7 },
					{ 11, 13, 17 },
				};

				var x = graph.Const (test);

				if (axis == null || axis >= -2 && axis < 2) {
					TFOutput y = graph.ReduceProd (x, axis: axis == null ? (TFOutput?)null : graph.Const(axis));

					TFTensor [] result = session.Run (new TFOutput [] { }, new TFTensor [] { }, new [] { y });

					object actual = result [0].GetValue ();
					TestUtils.MatrixEqual (expected, actual, precision: 8);
				} else {
					Assert.Throws<TFException> (() => {
						TFOutput y = graph.ReduceProd (x, axis: axis == null ? (TFOutput?)null : graph.Const (axis));
						session.Run (new TFOutput [] { }, new TFTensor [] { }, new [] { y });
					});
				}
			}
		}

	}
}
