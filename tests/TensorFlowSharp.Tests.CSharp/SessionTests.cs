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
	}
}