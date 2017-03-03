//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;
using System.Runtime.InteropServices;

namespace TensorFlow
{
	using static NativeMethods;

	/// <summary>
	/// Represents a specific input of an operation.
	/// </summary>
	[StructLayout (LayoutKind.Sequential)]
	public struct TFInput
	{
		public IntPtr Operation;
		public int Index;


		public TFOutput GetOutput (TFInput operIn)
		{
			return TF_OperationInput (operIn);
		}

		public TFDataType InputType => TF_OperationInputType (this);

	}

}
