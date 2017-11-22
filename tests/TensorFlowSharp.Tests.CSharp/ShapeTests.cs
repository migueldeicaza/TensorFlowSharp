using System;
using System.Collections.Generic;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class ShapeTests
	{
		[Fact]
		public void Should_ShapeAutomaticallyConvertToTensor ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {

				var x = graph.Const (new TFShape(2, 3));

				TFTensor [] result = session.Run (new TFOutput [] { }, new TFTensor [] { }, new TFOutput [] { x });

				int[] actual = (int[])result [0].GetValue ();
				Assert.Equal (new [] { 2, 3 }, actual);
			}
		}

	}
}
