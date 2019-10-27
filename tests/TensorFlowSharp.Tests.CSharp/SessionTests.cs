using System;
using System.Linq;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class SessionTests
	{
		[Fact]
		public void Should_ListDeviceReturnDevices ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var devices = session.ListDevices ();

				Assert.True(devices.Any());
			}
		}
		
		[Theory]
		[InlineData("Placeholder")]
		[InlineData("Placeholder:0")]
		public void ParseOutput_ThrowsForMissingOp (string name)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph))
			{
				var runner = session.GetRunner();
				Assert.Throws<ArgumentOutOfRangeException>(() => runner.AddInput(name, new TFTensor(1)));
			}
		}
	}
}