//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorFlow
{
	using static NativeMethods;

	/// <summary>
	/// Represents a computation graph.  Graphs may be shared between sessions and are thread safe.
	/// </summary>
	/// <remarks>
	/// Graphs consist of operations (represented by TFOperation objects), these can be named, or 
	/// the runtime will automatically assign a name.
	/// 
	/// For debugging purposes, you might want to group operations together, for this, call the
	/// WithScope method with your new scope, which will create a new namespace for your object names.
	/// 
	/// For example, if you call WithScope ("demo"), and add an operation named "add" inside the
	/// scope, the full name of the operation will be "demo/add", if you create a new scope inside, say
	/// "hot", and add a "sub" operation there the result will be "demo/hot/sub".
	/// </remarks>
	public partial class TFGraph : TFDisposable
	{


		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
		/// </summary>
		public TFGraph () : base (TF_NewGraph ())
		{
		}

		internal TFGraph (IntPtr handle) : base (handle)
		{
		}


		protected override void NativeDispose (IntPtr handle)
		{
			TF_DeleteGraph (handle);
		}


		public void SetTensorShape (TFOutput output, long [] dims, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			if (dims == null)
				TF_GraphSetTensorShape (Handle, output, IntPtr.Zero, 0, cstatus.Handle);
			else
				TF_GraphSetTensorShape (Handle, output, ref dims, dims.Length, cstatus.Handle);

			cstatus.CheckMaybeRaise (status);
		}



		public int GetTensorNumDims (TFOutput output, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var code = TF_GraphGetTensorNumDims (Handle, output, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return code;
		}



		public long [] GetTensorShape (TFOutput output, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var n = TF_GraphGetTensorNumDims (Handle, output, cstatus.Handle);
			if (!cstatus.CheckMaybeRaise (status, false))
				return null;

			var dims = new long [n];

			TF_GraphGetTensorShape (Handle, output, ref dims, dims.Length, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return dims;
		}


		public void ToGraphDef (TFBuffer outputGraphDef, TFStatus status = null)
		{
			CheckDisposed ();

			if (outputGraphDef == null)
				throw new ArgumentNullException (nameof (outputGraphDef));

			var cstatus = TFStatus.Setup (status);
			unsafe
			{
				TF_GraphToGraphDef (Handle, outputGraphDef.LLBuffer, cstatus.Handle);
			}
			cstatus.CheckMaybeRaise (status);
		}


		public void Import (TFBuffer graphDef, string prefix = "", TFStatus status = null)
		{
			CheckDisposed ();

			if (graphDef == null)
				throw new ArgumentNullException (nameof (graphDef));
			if (prefix == null)
				throw new ArgumentNullException (nameof (prefix));

			using (var options = new TFImportGraphDefOptions ()) {
				options.SetPrefix (prefix);
				Import (graphDef, options, status);
			}
		}

		public void Import (TFBuffer graphDef, TFImportGraphDefOptions options, TFStatus status = null)
		{
			CheckDisposed ();

			if (graphDef == null)
				throw new ArgumentNullException (nameof (graphDef));
			if (options == null)
				throw new ArgumentNullException (nameof (options));

			var cstatus = TFStatus.Setup (status);
			unsafe
			{
				TF_GraphImportGraphDef (Handle, graphDef.LLBuffer, options.Handle, cstatus.Handle);
			}
			cstatus.CheckMaybeRaise (status);
		}

		public void Import (byte [] buffer, string prefix = "", TFStatus status = null)
		{
			CheckDisposed ();

			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));
			if (prefix == null)
				throw new ArgumentNullException (nameof (prefix));
			using (var options = new TFImportGraphDefOptions ()) {
				options.SetPrefix (prefix);
				Import (buffer, options, status);
			}
		}

		public void Import (byte [] buffer, TFImportGraphDefOptions options, TFStatus status = null)
		{
			CheckDisposed ();

			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));
			if (options == null)
				throw new ArgumentNullException (nameof (options));
			var cstatus = TFStatus.Setup (status);
			using (var tb = new TFBuffer (buffer, 0, buffer.Length))
				Import (tb, options, status);

			cstatus.CheckMaybeRaise (cstatus);
		}


		public TFOperation this [string name] {
			get {
				CheckDisposed ();

				var h = TF_GraphOperationByName (Handle, name);
				if (h == IntPtr.Zero)
					return null;
				return new TFOperation (this, h);
			}
		}


		public IEnumerable<TFOperation> GetEnumerator ()
		{
			CheckDisposed ();

			var token = IntPtr.Zero;
			IntPtr operll;

			while ((operll = TF_GraphNextOperation (Handle, ref token)) != IntPtr.Zero)
				yield return new TFOperation (this, operll);
		}

		/// <summary>
		///  Returns the tensor shape for the specific output pparameters as an array of longs.
		/// </summary>
		/// <returns>null for single dimension, .</returns>
		/// <param name="output">The output operation to probe.</param>
		/// <param name="status">Status.</param>
		public long [] GetShape (TFOutput output, TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			var ndims = TF_GraphGetTensorNumDims (Handle, output, cstatus.Handle);
			if (!cstatus.CheckMaybeRaise (status, false))
				return null;

			if (ndims == 0)
				return null;

			var ret = new long [ndims];
			TF_GraphGetTensorShape (Handle, output, ref ret, ndims, cstatus.Handle);

			cstatus.CheckMaybeRaise (status);
			return ret;
		}

		/// <summary>
		/// Returns the current name scope in use, to change this, use the WithScope method.
		/// </summary>
		/// <value>The current name scope.</value>
		public string CurrentNameScope { get; internal set; } = string.Empty;

		/// <summary>
		/// Creates a new namescope by setting the scope to the description provided.
		/// </summary>
		/// <returns>A new scope that will remain in use until the return TFScope is disposed.</returns>
		/// <param name="nameScopeDesc">The namescope description, if the value is null, this
		/// will reset the toplevel namescope to be the empty value. </param>
		/// <remarks>
		/// To more easily name your operations and group then, you can use the
		/// WithScope method to set a current name scope that alter the complete name
		/// of an operation added to the graph.
		/// 
		/// The graph starts with a scope set to the empty string, you can introduce new
		/// scopes by calling WithScope, and can be conveniently used with the C# using
		/// statement, like this:
		/// 
		/// <code>
		/// Assert (graph.CurrentNamescope, "");
		/// using (var nested = graph.WithScope ("nested")){
		///    Assert (graph.CurrentNameScope, "nested");
		///    using (var inner = graph.WithScope ("inner")){
		///        Assert (graph.CurrentNameScope, "nested/inner");
		///    }
		/// }
		/// </code>
		/// </remarks>
		public TFScope WithScope (string nameScopeDesc)
		{
			var scope = new TFScope (this);
			if (scope == null)
				CurrentNameScope = "";
			else if (CurrentNameScope.Length == 0)
				CurrentNameScope = nameScopeDesc;
			else
				CurrentNameScope = CurrentNameScope + "/" + nameScopeDesc;

			return scope;
		}

	    private Dictionary<string, int> values = new Dictionary<string, int> ();

	    private string MakeName (string operName, string userName)
		{
			if (userName == null) {
				var k = CurrentNameScope == "" ? operName : CurrentNameScope + "/" + operName;

				return MakeUnique (k);
			}
			if (CurrentNameScope == "")
				return userName;
			return CurrentNameScope + "/" + userName;
		}

	    private string MakeUnique (string name)
		{
			var val = 0;

			if (!values.TryGetValue (name, out val))
				val = 0;
			else
				val++;
			values [name] = val;
			return name + val;
		}


		/// <summary>
		/// Imports a graph serialized into the graph
		/// </summary>
		/// <param name="graphDef">Serialized graph definition (in protocol buffer format).</param>
		/// <param name="options">Import options.</param>
		/// <param name="returnOutputs">Array large enough to contain all the return options.</param>
		/// <param name="status">Status, optional.</param>
		public void ImportGraphDef (TFBuffer graphDef, TFImportGraphDefOptions options, TFOutput [] returnOutputs, TFStatus status = null)
		{
			CheckDisposed ();

			if (graphDef == null)
				throw new ArgumentNullException (nameof (graphDef));
			if (options == null)
				throw new ArgumentNullException (nameof (options));
			var cstatus = TFStatus.Setup (status);

			unsafe
			{
				if (returnOutputs == null) {
					TF_GraphImportGraphDefWithReturnOutputs (Handle, graphDef.LLBuffer, options.Handle, null, 0, cstatus.Handle);
				} else {
					fixed (TFOutput* first = &returnOutputs [0]) {
						TF_GraphImportGraphDefWithReturnOutputs (Handle, graphDef.LLBuffer, options.Handle, first, returnOutputs.Length, cstatus.Handle);
					}
				}
			}
		}

		private static unsafe TFOutput [] CopyFrom (TFOutput* ptr, int n)
		{
			var r = new TFOutput [n];
			for (var i = 0; i < n; i++)
				r [i] = ptr [i];

			return r;
		}

		/// <summary>
		/// Signature of the method that will be invoked by the TFGraph.While method to construct a while loop
		/// </summary>
		/// <remarks>
		/// The method should build up the condition on the conditionGraph and the body of the while 
		/// loop in the provided bodyGraph.   It should set the condOutput to the value used as the
		/// condition output and the array of values in bodyOutputs to the final outputs as well as the
		/// name to be used, if not set, one will be assigned.
		/// 
		/// The conditionGraph represents the while condition and the inputs are the current values of the
		/// input variables (condInputs).   The output should be a scalar boolean.
		/// 
		/// The loop body graph is in bodyGraph, The inputs are the current values of the loop
		/// variables. The outputs are the updated values of the loop variables.
		/// 
		/// You can use the passed status record problems with it.
		/// </remarks>
		public delegate void WhileConstructor (TFGraph conditionGraph, TFOutput [] condInputs, out TFOutput condOutput, TFGraph bodyGraph, TFOutput [] bodyInputs, TFOutput [] bodyOutputs, out string name);

		/// <summary>
		/// Constructs a while loop with the specified inputs and a callback that composes the while loop
		/// </summary>
		/// <param name="inputs">Inputs.</param>
		/// <param name="whileConstructor">Callback method that fills out the various while loop parameters.</param>
		/// <returns>
		/// An array of TFOutputs from creating the While loop, or null if there is an error creating the 
		/// while loop, or if the constructor raised an exception when it was invoked.
		/// </returns>
		public TFOutput [] While (TFOutput [] inputs, WhileConstructor whileConstructor, TFStatus status = null)
		{
			CheckDisposed ();

			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (whileConstructor == null)
				throw new ArgumentNullException (nameof (whileConstructor));

			var cstatus = TFStatus.Setup (status);
			var result = TF_NewWhile (Handle, inputs, inputs.Length, cstatus.Handle);

			if (cstatus.Error)
				return null;

			try {

				// 
				// Call constructor here
				// Wrap the various TF_graphs (with owns=false)
				// Marshal the condInputs, bodyInputs
				//
				string name;

				var n = result.ninputs;
				var bodyOutputs = new TFOutput [n];
				unsafe
				{
					var condGraph = new TFGraphUnowned (result.cond_graph);
					var bodyGraph = new TFGraphUnowned (result.body_graph);
					whileConstructor (condGraph, CopyFrom (result.cond_inputs, n), out result.cond_output, bodyGraph, CopyFrom (result.body_inputs, n), bodyOutputs, out name);
				}
				if (string.IsNullOrEmpty (name))
					name = MakeUnique ("while");

				// On return, copy the condOutput and bodyOututs
				var text = Encoding.UTF8.GetBytes (name);

				result.charPtrName = Marshal.AllocHGlobal (text.Length + 1);
				Marshal.Copy (text, 0, result.charPtrName, text.Length);
				Marshal.WriteByte (result.charPtrName, text.Length, 0);

				unsafe
				{
					for (var i = 0; i < n; i++)
						result.body_outputs [i] = bodyOutputs [i];
					var ret = new TFOutput [inputs.Length];
					fixed (TFOutput* first = &ret [0])
						TF_FinishWhile (ref result, cstatus.Handle, first);


					if (cstatus.CheckMaybeRaise (status))
						return ret;
				}
				return null;
			} catch {
				TF_AbortWhile (ref result);
				return null;
			}
		}

		/// <summary>
		/// Creates a constant operation from a TFTensor or constant
		/// </summary>
		/// <param name="value">Value.</param>
		/// <param name="operName">Oper name.</param>
		/// <remarks>
		/// Since TFTensor have implicit conversion operators, you can call this method with
		/// a constant like this: graph.Const (23)
		/// </remarks>
		public TFOutput Const (TFTensor value, string operName = null)
		{
			return Const (value, value.TensorType, operName);
		}

		// Returns range(0, rank(x)) if reduction_indices is null
	    private TFOutput ReduceDims (TFOutput input, TFOutput? axis = null)
		{
			if (axis.HasValue)
				return axis.Value;

			// Fast path: avoid creating Rank and Range ops if ndims is known.
			var shape = GetTensorShape (input);
			if (shape.Length >= 0) {
				// The python code distinguishes between tensor and sparsetensor

				var array = new int [shape.Length];
				for (var i = 0; i < array.Length; i++)
					array [i] = i;

				return Const (array, TFDataType.Int32);
			}
			return Range (Const (0), Const (shape.Length), Const (1));
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
		public TFOutput ReduceSum (TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
		{
			return Sum (input, ReduceDims (input, axis), keep_dims, operName);
		}

		/// <summary>
		/// Variable node, with a starting initial value.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="init">Returns the operation that initializes the value of the variable.</param>
		/// <param name="value">Returns the value of the variable.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning TFOutput returns the handle to the variable.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable
		/// </remarks>
		public TFOutput Variable (TFOutput initialValue, out TFOperation init, out TFOutput value, string operName = null)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var handle = VarHandleOp (type, new TFShape (GetShape (initialValue)));
				using (var aScope = WithScope ("Assign")) {
					init = AssignVariableOp (handle, initialValue);
					using (var rScope = WithScope ("Read")) {
						value = ReadVariableOp (handle, type);
						return handle;
					}
				}
			}
		}

	    private List<TFOperation> pending_init_variables;
		public void AddInitVariable (TFOperation variable)
		{
			if (pending_init_variables == null)
				pending_init_variables = new List<TFOperation> ();
			pending_init_variables.Add (variable);
		}

		public TFOperation [] GetGlobalVariablesInitializer ()
		{
			var res = pending_init_variables.ToArray ();
			pending_init_variables.Clear ();
			return res;
		}

		/// <summary>
		/// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="value">Returns the value of the variable.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning TFOutput returns the handle to the variable.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable.
		/// 
		/// The init sequence for the variable is stored in the graph, you must manually initialize 
		/// those by running the session on the global variables.
		/// </remarks>
		public TFOutput Variable (TFOutput initialValue, out TFOutput value, string operName = null)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var handle = VarHandleOp (type, new TFShape (GetShape (initialValue)));
				using (var aScope = WithScope ("Assign")) {
					var init = AssignVariableOp (handle, initialValue);
					AddInitVariable (init);
					using (var rScope = WithScope ("Read")) {
						value = ReadVariableOp (handle, type);
						return handle;
					}
				}
			}
		}

		/// <summary>
		/// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning TFOutput returns the handle to the variable.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable.
		/// 
		/// The init sequence for the variable is stored in the graph, you must manually initialize 
		/// those by running the session on the global variables.
		/// </remarks>
		public TFOutput Variable (TFOutput initialValue, string operName = null)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var handle = VarHandleOp (type, new TFShape (GetShape (initialValue)));
				using (var aScope = WithScope ("Assign")) {
					var init = AssignVariableOp (handle, initialValue);
					AddInitVariable (init);
					return handle;
				}
			}
		}

	}
}
