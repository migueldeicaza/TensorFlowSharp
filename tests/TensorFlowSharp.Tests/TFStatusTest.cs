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
	public class TFStatusTest
	{
		[Test]
		public void DefaultTest ()
		{

			using (var status = new TFStatus ()) {

				Assert.AreEqual (TFStatusCode.Ok, status.Code);
			
				Assert.IsEmpty (status.Message);

				status.SetStatus (TFStatusCode.Cancelled, "cançel");

				Assert.AreEqual (TFStatusCode.Cancelled, status.Code);

				Assert.AreEqual ("cançel", status.Message);

			}
		}
	}
}
