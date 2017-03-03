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
	//
	// A TFGraph that will not release the undelying handle, this is used
	// when we want to surface a TFGraph that we do not own, so we do not
	// want to delete the handle when this object is collected
	//
	internal class TFGraphUnowned : TFGraph
	{
		internal TFGraphUnowned (IntPtr handle) : base (handle)
		{
		}

		protected override void NativeDispose (IntPtr handle)
		{
			// nothing, we do not own the handle
		}
	}
}
