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
	public class TFTensorTest
	{
		[Test]
		public void DefaultFloatTest ()
		{
			var value = 1f;
			using (var tensor = new TFTensor (value)) {
				Assert.AreEqual (0, tensor.NumDims, "The default constructor must initialize with zero dimentions.");
				Assert.AreEqual (value, tensor.GetValue ());
				Assert.AreEqual (TFDataType.Float, tensor.TensorType);
			}
		}
		[Test]
		public void DefaultInt32Test ()
		{
			var value = 85;
			using (var tensor = new TFTensor (value)) {
				Assert.AreEqual (0, tensor.NumDims, "The default constructor must initialize with zero dimentions.");
				Assert.AreEqual (value, tensor.GetValue ());
				Assert.AreEqual (TFDataType.Int32, tensor.TensorType);
			}
		}
	}
}
