//
// TensorFlow API for .NET languages
// 
// Authors:
//   Miguel de Icaza (miguel@microsoft.com)
//   Gustavo J Knuppe (https://github.com/knuppe/)
//
// Strongly typed API
// The API generally takes a TF_Status that defaults to null, if the value is null, on error, this raises an exception, otherwise, the error is returned on the TF_Status.
// You can use TFStatus.Default for a value to use when you do not want to create the value yourself and are OK reusing the value.
//
// Guidance on doing language bindings for TensorFlow:
// https://www.tensorflow.org/versions/r0.11/how_tos/language_bindings/
//
//
namespace TensorFlow
{
    using static NativeMethods;

	/// <summary>
	/// TensorFlow Core Library
	/// </summary>
	public static class TFCore
	{

		/// <summary>
		/// Gets the TensorFlow version.
		/// </summary>
		/// <value>The TensorFlow version.</value>
		public static string Version => TF_Version ().GetStr ();

		public static long GetDataTypeSize (TFDataType dt) => (long)TF_DataTypeSize (dt);

		public static TFBuffer GetAllOpList ()
		{
			return new TFBuffer (TF_GetAllOpList ());
		}
	}
}