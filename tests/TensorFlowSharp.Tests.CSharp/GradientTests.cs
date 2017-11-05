using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class GradientTests
	{
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
	}
}
