using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class GradientTests
	{
        private const float _tolerance = 0.000001f;
        [Fact]
		public void ShouldAddGradients ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var x = graph.Const (3.0);

				var y1 = graph.Square (x, "Square1");
				var y2 = graph.Square (y1, "Square2");

				var y3 = graph.Square (y2, "Square3");
				var g = graph.AddGradients (new TFOutput [] { y1, y3 }, new [] { x });

				var r = session.Run (new TFOutput [] { }, new TFTensor [] { }, g);
				var dy = (double)r [0].GetValue ();
				Assert.Equal (17502.0, dy);
			}
		}


        [Fact]
        public void ComputeGradientMSE()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var X = graph.Const(5.5f);
                var Y = graph.Const(2.09f);

                var W = graph.Const(0.1078f);
                var b = graph.Const(0.1021f);
                var pred = graph.Add(graph.Mul(X, W, "x_w"), b);

                var cost = graph.Div(graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2f))), graph.Mul(graph.Const(2f), graph.Const((float)17), "2_n_samples"));

                var g = graph.AddGradients(new TFOutput[] { cost }, new[] { W });

                var r = session.Run(new TFOutput[] { }, new TFTensor[] {  }, new TFOutput[] { cost, g[0] });
                var d = (float)r[0].GetValue();
                Assert.InRange(d, 0.057236027 - _tolerance, 0.057236027 + _tolerance);
                d = (float)r[1].GetValue();
                Assert.InRange(d, -0.4513235 - _tolerance, -0.4513235 + _tolerance);
            }
        }
    }
}
