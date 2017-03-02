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
	/// <summary>
	/// TFGraph name scope handle
	/// </summary>
	/// <remarks>
	/// Instances of this class when disposed restore the CurrentNameScope to the
	/// value they had when the TFGraph.WithScope method was called.
	/// </remarks>
	public class TFScope : IDisposable
	{
		private readonly TFGraph container;
		private readonly string name;

		internal TFScope (TFGraph container)
		{
			this.container = container;
			name = container.CurrentNameScope;
		}

		public void Dispose ()
		{
			container.CurrentNameScope = name;
		}
	}
}
