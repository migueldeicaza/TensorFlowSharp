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

	public class TFSessionOptions : TFDisposable
	{
		public TFSessionOptions () : base (TF_NewSessionOptions ()) { }

		protected override void NativeDispose (IntPtr handle)
		{
			TF_DeleteSessionOptions (handle);
		}

		public void SetTarget (string target)
		{
			CheckDisposed ();

			TF_SetTarget (Handle, target);
		}


		public void SetConfig (IntPtr protoData, int length, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);

			TF_SetConfig (Handle, protoData, (UIntPtr)length, cstatus.Handle);

			cstatus.CheckMaybeRaise (status);
		}

	}
}
