using System;
namespace TensorFlow
{
	public partial class TFGraph
	{
		/// <summary>
		/// Creates a constant from a TFTensor
		/// </summary>
		/// <param name="value">Value.</param>
		/// <param name="operName">Oper name.</param>
		public TFOutput Const (TFTensor value, string operName = null)
		{
			return Const (value, value.TensorType, operName);
		}
	}
}
