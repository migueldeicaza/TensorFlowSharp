//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System.Runtime.InteropServices;

namespace TensorFlow
{

	[StructLayout (LayoutKind.Sequential)]
	public struct TFAttributeMetadata
	{
	    private byte isList;
		public bool IsList => isList != 0;
		public long ListSize;
		public TFAttributeType Type;
		public long TotalSize;

		public override string ToString ()
		{
			return string.Format ($"[TFAttributeMetadata IsList={IsList} ListSize={ListSize} Type={Type} TotalSize={TotalSize}]");
		}
	}
}
