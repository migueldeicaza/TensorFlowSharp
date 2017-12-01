using System.Collections.Generic;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class CondTests
    {
		[Theory]
		[InlineData (false)]
		[InlineData (true)]
		public void Should_ExecuteOnlyOne (bool flag)
		{
			using (var graph = new TFGraph ()) {

				var W = graph.VariableV2 (TFShape.Scalar, TFDataType.Double, operName: "W");

				var b = graph.VariableV2 (TFShape.Scalar, TFDataType.Double, operName: "b");

				var pred = graph.Const (flag);

				var init = graph.Cond (pred, 
					() => graph.Assign (W, graph.Const (1.0)),
					() => graph.Assign (b, graph.Const (-0.3)));

				using (var sess = new TFSession (graph)) {

					Assert.Throws<TFException> (() => sess.GetRunner ().Fetch (W).Run ()); // ok
					Assert.Throws<TFException> (() => sess.GetRunner ().Fetch (b).Run ()); // ok

					var r1 = sess.GetRunner ().AddTarget (init.Operation).Run ();

					if (flag) {
						var rW = sess.GetRunner ().Fetch (W).Run ();
						Assert.Throws<TFException> (() => sess.GetRunner ().Fetch (b).Run ());
						Assert.Equal (1.0, (double)rW [0].GetValue ());
					} else {
						Assert.Throws<TFException> (() => sess.GetRunner ().Fetch (W).Run ());
						var rb = sess.GetRunner ().Fetch (b).Run ();
						Assert.Equal (-0.3, (double)rb [0].GetValue ());
					}
					
				}
			}
		}

    }
}
