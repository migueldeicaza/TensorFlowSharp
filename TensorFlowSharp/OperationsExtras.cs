using System;
using System.Linq;

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

		// Returns range(0, rank(x)) if reduction_indices is null
		TFOutput ReduceDims (TFTensor input, TFOutput? axis = null)
		{
			if (axis.HasValue)
				return axis.Value;

			// Fast path: avoid creating Rank and Range ops if ndims is known.
			if (input.NumDims >= 0) {
				// The python code distinguishes between tensor and sparsetensor

				var array = new int [input.NumDims];
				for (int i = 0; i < array.Length; i++)
					array [i] = i;

				return this.Const (array, TFDataType.Int32);                   
			}
			return Range (Const (0), Const (input.NumDims), Const (1));
		}

		/// <summary>
		/// Computes the sum of elements across dimensions of a tensor.
		/// </summary>
		/// <returns>The reduced tensor.</returns>
		/// <param name="input">The tensor to reduce. Should have numeric type.</param>
		/// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
		/// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
		/// <param name="operName">A name for the operation, optional.</param>
		/// <remarks>
		///   Reduces input_tensor along the dimensions given in axis.
		/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
		/// entry in axis. If keep_dims is true, the reduced dimensions
		/// are retained with length 1.
		/// 
		/// If axis has no entries, all dimensions are reduced, and a
		/// tensor with a single element is returned.
		/// </remarks>
		public TFOutput ReduceSum (TFTensor input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
		{
			return Sum (Const (input), this.ReduceDims (input, axis), keep_dims, operName);
		}
	}
}
