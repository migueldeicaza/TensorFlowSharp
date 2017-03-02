//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

namespace TensorFlow
{

	/// <summary>
	/// The data type for a specific tensor.
	/// </summary>
	/// <remarks>
	/// Tensors have uniform data types, all the elements of the tensor are of this
	/// type and they dictate how TensorFlow will treat the data stored.   
	/// </remarks>
	public enum TFDataType : uint
	{
		Float = 1,
		Double = 2,
		Int32 = 3,
		UInt8 = 4,
		Int16 = 5,
		Int8 = 6,
		String = 7,
		Complex64 = 8,
		Complex = 8,
		Int64 = 9,
		Bool = 10,
		QInt8 = 11,
		QUInt8 = 12,
		QInt32 = 13,
		BFloat16 = 14,
		QInt16 = 15,
		QUInt16 = 16,
		UInt16 = 17,
		Complex128 = 18,
		Half = 19,
		Resource = 20
	}

}
