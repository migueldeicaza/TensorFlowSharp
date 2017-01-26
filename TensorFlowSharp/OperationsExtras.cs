using System;
namespace TensorFlow
{
	public partial class TFGraph
	{
		
		public TFOutput Const (Scope scope, TFTensor value, string operName = null)
		{
			return Const (scope, value, value.TensorType, operName);
		}

	}
}
