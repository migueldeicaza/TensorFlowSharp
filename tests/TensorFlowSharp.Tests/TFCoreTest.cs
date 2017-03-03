//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//
using NUnit.Framework;

using TensorFlow;

namespace TensorFlowSharp.Tests
{
	[TestFixture]
	public class TFCoreTest
	{
		[Test]
		public void VersionTest ()
		{
			var version = TFCore.Version;

			Assert.NotNull (version);
			Assert.IsNotEmpty (version);

		}
	}
}
