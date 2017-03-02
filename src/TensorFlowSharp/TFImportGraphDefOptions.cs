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

	public class TFImportGraphDefOptions : TFDisposable
	{


		public TFImportGraphDefOptions () : base (TF_NewImportGraphDefOptions ())
		{
		}


		protected override void NativeDispose (IntPtr handle)
		{
			TF_DeleteImportGraphDefOptions (handle);
		}


		public void SetPrefix (string prefix)
		{
			CheckDisposed ();

			TF_ImportGraphDefOptionsSetPrefix (Handle, prefix);
		}



		/// <summary>
		/// Adds an input mapping from a source name and index to a destination output
		/// </summary>
		/// <param name="srcName">Source name.</param>
		/// <param name="srcIndex">Source index (in the source).</param>
		/// <param name="dst">Replacement value for the srcName:srcIndex.</param>
		/// <remarks>
		/// Set any imported nodes with input `src_name:src_index` to have that input
		/// replaced with `dst`. `src_name` refers to a node in the graph to be imported,
		/// `dst` references a node already existing in the graph being imported into.
		/// </remarks>
		public void AddInputMapping (string srcName, int srcIndex, TFOutput dst)
		{
			CheckDisposed ();
			TF_ImportGraphDefOptionsAddInputMapping (Handle, srcName, srcIndex, dst);
		}



		/// <summary>
		/// Cause the imported graph to have a control dependency on the provided operation.
		/// </summary>
		/// <param name="operation">This operation should exist in the graph being imported to.</param>
		public void AddControlDependency (TFOperation operation)
		{
			if (operation == null)
				throw new ArgumentNullException (nameof (operation));
			
			CheckDisposed ();

			TF_ImportGraphDefOptionsAddControlDependency (Handle, operation.Handle);
		}	


		/// <summary>
		/// Add an output in the graph definition to be returned via the return outputs parameter.
		/// </summary>
		/// <param name="operName">Operation name.</param>
		/// <param name="index">Operation index.</param>
		/// <remarks>
		/// If the output is remapped via an input
		/// mapping, the corresponding existing tensor in graph will be returned.
		/// </remarks>
		public void AddReturnOutput (string operName, int index)
		{
			if (operName == null)
				throw new ArgumentNullException (nameof (operName));
			
			CheckDisposed ();
			TF_ImportGraphDefOptionsAddReturnOutput (Handle, operName, index);
		}



		/// <summary>
		/// Gets the number return outputs added via AddReturnOutput.
		/// </summary>
		/// <value>The number return outputs.</value>
		public int NumReturnOutputs {
			get {
				CheckDisposed ();

				return TF_ImportGraphDefOptionsNumReturnOutputs (Handle);
			}
		}

	}
}
