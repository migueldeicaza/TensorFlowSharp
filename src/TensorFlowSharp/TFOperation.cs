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

	/// <summary>
	/// Represents a computation node in the graph. Tensorflow operations are attached to a <see cref="T:TFGraph"/>.
	/// </summary>
	/// <remarks>
	/// TFOperations are usually created by  invoking one of the methods in 
	/// <see cref="T:TFGraph"/>, but they can also be constructed manually 
	/// using the low-level <see cref="T:TFOperationDesc"/> API.
	/// </remarks>
	public class TFOperation : TFDisposable
	{

		// Pointer to the graph, to keep it from collecting if there are TFOperations alive.
		internal TFGraph graph;

		internal TFOperation (TFGraph graph, IntPtr handle) : base (handle)
		{
			this.graph = graph;
		}

		/// <summary>
		/// The name for this operation/
		/// </summary>
		/// <value>The name.</value>
		public string Name => IsDisposed ? "<ObjectDisposed>" : TF_OperationName (Handle).GetStr ();

		public string OpType => IsDisposed ? "<ObjectDisposedException>" : TF_OperationOpType (Handle).GetStr ();

	
		/// <summary>
		/// Gets the number of outputs on this operation.
		/// </summary>
		/// <value>The number outputs.</value>
		public int NumOutputs => Handle == IntPtr.Zero ? -1 : TF_OperationNumOutputs (Handle);


		protected override void NativeDispose (IntPtr handle)
		{
			// nothing, we do not own the handle
		}

		public int OutputListLength (string argName, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var res = TF_OperationOutputListLength (Handle, argName, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return res;
		}

		/// <summary>
		/// Gets the number of inputs for this operation.
		/// </summary>
		/// <value>The number inputs.</value>
		public int NumInputs => TF_OperationNumInputs (Handle);

		// public string Device => TF_OperationDevice (handle).GetStr ();

		public int InputListLength (string argName, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var res = TF_OperationInputListLength (Handle, argName, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return res;
		}


		public int NumControlInputs => TF_OperationNumControlInputs (Handle);


		public int NumControlOutputs => TF_OperationNumControlOutputs (Handle);

	    private TFOperation [] ControlOutputs {
			get {
				var n = NumControlOutputs;
				var arr = new IntPtr [n];
				TF_OperationGetControlOutputs (Handle, arr, n);
				var ret = new TFOperation [n];
				for (var i = 0; i < n; i++)
					ret [i] = new TFOperation (graph, arr [i]);
				return ret;
			}
		}


		public TFAttributeMetadata GetAttributeMetadata (string attrName, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var x = TF_OperationGetAttrMetadata (Handle, attrName, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return x;
		}


		/// <summary>
		/// Encodes the TFOperation as a protocol buffer payload
		/// </summary>
		/// <returns>The buffer with the encoded operation in the protocol buffer format.</returns>
		/// <param name="status">Status.</param>
		/// <remarks>
		/// </remarks>
		public TFBuffer ToNodeDef (TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var r = new TFBuffer ();
			unsafe
			{
				TF_OperationToNodeDef (Handle, r.LLBuffer, cstatus.Handle);
			}
			// No need to raise, we can return null in that case.
			if (!cstatus.IsOk) {
				r.Dispose ();
				return null;
			}
			return r;
		}

		public TFOutput this [int index] {
			get {
				CheckDisposed ();
				return new TFOutput (this, index);
			}
		}
	}
}
