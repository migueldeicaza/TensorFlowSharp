//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;

namespace TensorFlow
{
	using static NativeMethods;

	public sealed class TFLibrary : TFDisposable
	{
		private TFLibrary (IntPtr handle) : base (handle) { }

		public static TFLibrary FromFile (string libraryFile, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var h = TF_LoadLibrary (libraryFile, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return new TFLibrary (h);
		}

		protected override void NativeDispose (IntPtr handle)
		{
			TF_DeleteLibraryHandle (handle);
		}
	}
}
