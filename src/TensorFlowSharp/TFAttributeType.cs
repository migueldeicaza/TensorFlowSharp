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
	public enum TFAttributeType : uint
	{
		String = 0,
		Int = 1,
		Float = 2,
		Bool = 3,
		Type = 4,
		Shape = 5,
		Tensor = 6,
		Placeholder = 7,
		Func = 8
	}
}