using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class GradientTests
    {
		[Fact]
		public void Should_Tests ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var x = graph.Const (3);

				var y1 = graph.Square (x, "Square1");
				var y2 = graph.Square (y1, "Square2");

				var y3 = graph.Square (y2, "Square3");
				var g = graph.AddGradients (new TFOutput [] {y1,y2 }, new [] { x },new TFOutput [] { graph.Const(3) });

				var r = session.Run (new TFOutput [] { }, new TFTensor [] { }, g);
				int dy = (int)r [0].GetValue ();
				int dy2 = (int)r [1].GetValue ();
				Assert.Equal (17502.0, dy + dy2);
			}
		}
    }
}
