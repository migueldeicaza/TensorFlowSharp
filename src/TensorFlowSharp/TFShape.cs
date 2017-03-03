//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//


using System.Linq;

namespace TensorFlow
{

	/// <summary>
	/// Represents the shape of a tensor
	/// </summary>
	/// <remarks>
	/// The shapes can be created by calling the constructor with the number of dimensions
	/// in the shape.   The null value is used to specify that the shape is unknown,
	/// an empty array is used to create a scalar, and other values are used to specify
	/// the number of dimensions.
	/// 
	/// For the Unknown case, you can use <see cref="P:TensorFlor.TFShape.Unknown"/>, for
	/// scalars, you can use the <see cref="P:TensorFlor.TFShape.Scalar"/> shape.
	/// 
	/// To create a 2-element vector, use:
	/// new TFShape (2)
	/// 
	/// To create a 2x3 matrix, use:
	/// new TFShape (2, 3)
	/// 
	/// To create a shape with an unknown number of elements, you can pass the value
	/// -1.  This is typically used to indicate the shape of tensors that represent a
	/// variable-sized batch of values.
	/// 
	/// 
	/// To create a matrix with 4 columns and an unknown number of rows:
	/// var batch = new TFShape (-1, 4)
	/// </remarks>
	public class TFShape
	{
		/// <summary>
		/// Represents an unknown number of dimensions in the tensor.
		/// </summary>
		/// <value>The unknown.</value>
		public static TFShape Unknown => new TFShape (null);

		/// <summary>
		/// This shape is used to represent scalar values.
		/// </summary>
		/// <value>The scalar.</value>
		public static TFShape Scalar => new TFShape ();

		internal long [] dims;

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFShape"/> class.
		/// </summary>
		/// <param name="args">This is a params argument, so you can provide multiple values to it.  
		/// A null value means that this is an unknown shape, a single value is used to create a vector,
		/// two values are used to create a 2-D matrix and so on.
		/// </param>
		/// <remarks>
		/// 
		/// </remarks>
		public TFShape (params long [] args)
		{
			dims = args;
		}

		/// <summary>
		/// Gets the length of the specified dimension in the tensor
		/// </summary>
		/// <returns>The length, -1 for shapes that have an unknown dimension.</returns>
		/// <param name="dimension">Dimension.</param>
		public int GetLength (int dimension) => dims?.GetLength (dimension) ?? -1;

		/// <summary>
		/// Number of dimensions represented by this shape.
		/// </summary>
		/// <value>The number dimensions, -1 if the number of dimensions is unknown, 0 if the shape represent a scalar, 1 for a vector, 2 for a matrix and so on..</value>
		public int NumDimensions => dims?.Length ?? -1;

		/// <summary>
		/// Gets a value indicating whether all the dimensions in the <see cref="T:TensorFlow.TFShape"/> are fully specified.
		/// </summary>
		/// <value><c>true</c> if is fully specified; otherwise, <c>false</c>.</value>
		public bool IsFullySpecified {
			get {
				if (dims == null)
					return false;
				foreach (var j in dims)
					if (j == -1)
						return false;
				return true;
			}
		}

		/// <summary>
		/// Returns the shape as an array
		/// </summary>
		/// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
		public long [] ToArray ()
		{
		    var ret = (long []) dims?.Clone ();

			return ret;
		}

		public override string ToString ()
		{
			if (dims == null)
				return "unknown";
			return "[" + string.Join (", ", dims.Select (x => x == -1 ? "?" : x.ToString ())) + "]";
		}
	}



}
