//
// ArrayOps: support for manipulating tensors
//
// Authors:
//   Stephanus van Staden
//
// This is a port of the Python code in tensorflow
//
// 
using System;
namespace TensorFlow
{
	public partial class TFGraph
	{
		/// <summary>
		/// Outputs Zero values based on shape of tensor
		/// </summary>
		/// <param name="shape">Shape of the output tensor</param>
		/// <param name="dtype">Optional Type of the Zero value. Default: Double</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns></returns>
		public TFOutput Zeros (TFShape shape, TFDataType dtype = TFDataType.Double, string operName = null)
		{
			return Constant (0, shape, dtype, operName);
		}

		/// <summary>
		/// Outputs One values based on shape of tensor
		/// </summary>
		/// <param name="shape">Shape of the output tensor</param>
		/// <param name="dtype">Optional Type of the Zero value. Default: Double</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns></returns>
		public TFOutput Ones (TFShape shape, TFDataType dtype = TFDataType.Double, string operName = null)
		{
			return Constant (1, shape, dtype, operName);
		}

		/// <summary>
		/// Create a constant tensor based on a shape
		/// Used by Zeros and Ones
		/// </summary>
		/// <param name="value">Value for tensor</param>
		/// <param name="tfshape">Shape of the tensor</param>
		/// <param name="dtype">Optional Type of the Zero value. Default: Double</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns></returns>
		/// see https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/framework/constant_op.py
		public TFOutput Constant (object value, TFShape tfshape, TFDataType dtype = TFDataType.Double, string operName = null)
		{
            if (tfshape.NumDimensions <= 0)
            {
                TFTensor tensor = TFTensor.Create1DTensor(dtype, value);
                return Const(tensor, tensor.TensorType, operName);
            }
            //convert the .net type to relevant tensorflow type
            object dtvalue = TFTensor.FetchSimple (dtype, value);

			var shape = tfshape.ToArray ();
			var idx = new int [shape.Length];
			for (int i = 0; i < shape.Length; i++) {
				if (shape [i] > Int32.MaxValue)
					throw new ArgumentOutOfRangeException ("Shape can not be longer than 32 bits");
			}

			Array data = null;
			if (tfshape.IsLongArray) data = Array.CreateInstance (dtvalue.GetType (), tfshape.ToArray ());
			else data = Array.CreateInstance (dtvalue.GetType (), tfshape.ToIntArray ());

			TFTensor.Set (data, dtype, shape, idx, 0, value);

			TFTensor tensor_value = new TFTensor (data);
			return Const (tensor_value, tensor_value.TensorType, operName);
		}

	}
}
