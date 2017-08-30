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
	}
}
