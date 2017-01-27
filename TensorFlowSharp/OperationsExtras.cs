using System;
namespace TensorFlow
{
	public partial class TFGraph
	{
		public TFOutput Const (TFTensor value, string operName = null)
		{
			return Const (value, value.TensorType, operName);
		}
	}
}
