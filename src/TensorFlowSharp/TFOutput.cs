//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;
using System.Runtime.InteropServices;

namespace TensorFlow
{

	using static NativeMethods;

	/// <summary>
	/// Represents a specific output of an operation on a tensor.
	/// </summary>
	/// <remarks>
	/// TFOutput objects represent one of the outputs of an operation in the graph
	/// (TFGraph).  Outputs have a data type, and eventually a shape that you can 
	/// retrieve by calling the <see cref="M:TensorFlow.TFGraph.GetShape"/> method.
	/// 
	/// These can be passed as an input argument to a function for adding operations 
	/// to a graph, or to the TFSession's Run and GetRunner method as values to be
	/// fetched.
	/// </remarks>
	[StructLayout (LayoutKind.Sequential)]
	public struct TFOutput
	{
	    private IntPtr LLOperation;

		public int Index;


		/// <summary>
		/// Gets the number consumers.
		/// </summary>
		/// <value>The number consumers.</value>
		/// <remarks>
		/// This number can change when new operations are added to the graph.
		/// </remarks>
		public int NumConsumers => TF_OperationOutputNumConsumers (this);


		/// <summary>
		/// Gets the type of the output.
		/// </summary>
		/// <value>The type of the output.</value>
		public TFDataType OutputType => TF_OperationOutputType (this);

		/// <summary>
		/// Initializes a new TFOutput instance.
		/// </summary>
		/// <param name="operation">The operation to which to attach the output.</param>
		/// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
		public TFOutput (TFOperation operation, int index = 0)
		{
			if (operation == null)
				throw new ArgumentNullException (nameof (operation));
			LLOperation = operation.Handle;
			Index = index;
		}

		/// <summary>
		/// Initializes a new TFOutput instance from another TFOutput
		/// </summary>
		/// <param name="output">The other TFOutput that is having its operation attached.</param>
		/// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
		public TFOutput (TFOutput output, int index = 0)
		{
			if (output.LLOperation == IntPtr.Zero)
				throw new ArgumentException ("Outputs does not have a valid operation pointer", nameof (output));
			
			LLOperation = output.LLOperation;
			Index = index;
		}


		/// <summary>
		/// Get list of all current consumers of a specific output of an operation
		/// </summary>	
		/// <value>The output consumers.</value>
		/// <remarks>
		/// A concurrent modification of the graph can increase the number of consumers of
		/// an operation.
		/// This can return null if the TFOutput does not point to a valid object.
		/// </remarks>
		public TFInput [] OutputConsumers {
			get {
				var result = new TFInput [NumConsumers];
				unsafe
				{
					fixed (TFInput* first = &result [0])
					TF_OperationOutputConsumers (this, first, result.Length);
				}
				return result;
			}
		}

		/// <summary>
		/// The associated operation.
		/// </summary>
		/// <value>The operation.</value>
		public TFOperation Operation => new TFOperation (null, LLOperation);
		public override string ToString ()
		{
			return string.Format ("[TFOutput: LLOperation=0x{0:X} Index={1} Operation={2}]", (long) LLOperation, Index, Operation);
		}
	}

}
