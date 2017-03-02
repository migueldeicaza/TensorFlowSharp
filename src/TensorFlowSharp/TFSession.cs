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

namespace TensorFlow
{
    using static NativeMethods;

	/// <summary>
	/// Drives the execution of a graph
	/// </summary>
	/// <remarks>
	/// This creates a new context to execute a TFGraph.   You can use the 
	/// constructor to create an empty session, or you can load an existing
	/// model using the FromSAvedModel static method in this class.
	/// </remarks>
	public class TFSession : TFDisposable
	{


		public TFGraph Graph { get; private set; }

	    private TFSession (IntPtr handle, TFGraph graph) : base (handle)
		{
			Graph = graph;
		}

		public TFSession (TFGraph graph, TFSessionOptions sessionOptions, TFStatus status = null) 
		{
			Graph = graph;
			var cstatus = TFStatus.Setup (status);
			var h = TF_NewSession (graph.Handle, sessionOptions.Handle, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			Handle = h;
		}

		public TFSession (TFGraph graph, TFStatus status = null) 
		{
			Graph = graph;
			var cstatus = TFStatus.Setup (status);
			var empty = TF_NewSessionOptions ();
			var h = TF_NewSession (graph.Handle, empty, cstatus.Handle);
			TF_DeleteSessionOptions (empty);
			cstatus.CheckMaybeRaise (status);
			Handle = h;
		}

		public TFSession (TFStatus status = null) : this (new TFGraph (), status)
		{
		}

		public TFSession FromSavedModel (TFSessionOptions sessionOptions, TFBuffer runOptions, string exportDir, string [] tags, TFGraph graph, TFBuffer metaGraphDef, TFStatus status = null)
		{
			if (graph == null)
				throw new ArgumentNullException (nameof (graph));
			if (tags == null)
				throw new ArgumentNullException (nameof (tags));
			if (exportDir == null)
				throw new ArgumentNullException (nameof (exportDir));
			if (runOptions == null)
				throw new ArgumentNullException (nameof (runOptions));
			if (metaGraphDef == null)
				throw new ArgumentNullException (nameof (metaGraphDef));
			var cstatus = TFStatus.Setup (status);
			unsafe
			{
				var h = TF_LoadSessionFromSavedModel (sessionOptions.Handle, runOptions.LLBuffer, exportDir, tags, tags.Length, graph.Handle, metaGraphDef.LLBuffer, cstatus.Handle);

				if (cstatus.CheckMaybeRaise (status)) {
					return new TFSession (h, graph);
				}
			}
			return null;
		}

		public void CloseSession (TFStatus status = null)
		{
			CheckDisposed ();

			var cstatus = TFStatus.Setup (status);
			TF_CloseSession (Handle, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
		}
		public void DeleteSession (TFStatus status = null)
		{
			CheckDisposed ();
			var cstatus = TFStatus.Setup (status);
			TF_DeleteSession (Handle, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
		}

		protected override void NativeDispose (IntPtr handle)
		{
			using (var s = new TFStatus ()) {
				TF_DeleteSession (Handle, s.Handle);
			}
		}



		/// <summary>
		/// Use the runner class to easily configure inputs, outputs and targets to be passed to the session runner.
		/// </summary>
		/// <remarks>
		/// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
		/// to construct the parameters that will be passed to the TFSession.Run method.
		/// 
		/// Instances of this class are created by calling the GetRunner method on the TFSession.
		/// </remarks>
		public class Runner
		{
		    private List<TFOutput> inputs = new List<TFOutput> (), outputs = new List<TFOutput> ();
		    private List<TFTensor> inputValues = new List<TFTensor> ();
		    private List<TFOperation> targets = new List<TFOperation> ();
		    private TFSession session;

			internal Runner (TFSession session)
			{
				this.session = session;
			}

			/// <summary>
			/// Adds an input to the session
			/// </summary>
			/// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
			/// <param name="input">Incoming port.</param>
			/// <param name="value">Value to assing to the incoming port.</param>
			public Runner AddInput (TFOutput input, TFTensor value)
			{
				if (value == null)
					throw new ArgumentNullException (nameof (value));
				inputs.Add (input);
				inputValues.Add (value);
				return this;
			}

			/// <summary>
			/// Adds the specified operations as the ones to be retrieved.
			/// </summary>
			/// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
			/// <param name="targets">One or more targets.</param>
			public Runner AddTarget (params TFOperation [] targets)
			{
				foreach (var t in targets)
					this.targets.Add (t);
				return this;
			}

			public Runner AddTarget (params string [] targetNames)
			{
				foreach (var tn in targetNames)
					targets.Add (session.Graph [tn]);
				
				return this;
			}

			public Runner Fetch (string operation, int index = 0)
			{
				var op = session.Graph [operation];
				outputs.Add (op [index]);
				return this;
			}

			public Runner Fetch (TFOutput output)
			{
				outputs.Add (output);
				return this;
			}

			public Runner Fetch (params TFOutput [] outputs)
			{
				foreach (var output in outputs)
					this.outputs.Add (output);
				return this;
			}

			public TFBuffer RunMetadata, RunOptions;

			public TFTensor [] Run (TFStatus status = null)
			{
				return session.Run (inputs.ToArray (), inputValues.ToArray (), outputs.ToArray (), targets.ToArray (), RunMetadata, RunOptions, status);
			}

			/// <summary>
			/// Run the specified operation, by adding it implicity to the output, single return value
			/// </summary>
			/// <param name="operation">The output of the operation.</param>
			/// <param name="status">Optional, status.</param>
			public TFTensor [] Run (TFOutput operation, TFStatus status = null)
			{
				Fetch (operation);
				return Run (status);
			}

		}

		/// <summary>
		/// Gets a new runner, this provides a simpler API to prepare the inputs to run on a session
		/// </summary>
		/// <returns>The runner.</returns>
		/// <remarks>
		/// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
		/// to construct the parameters that will be passed to the TFSession.Run method.
		/// </remarks>
		public Runner GetRunner ()
		{
			return new Runner (this);
		}

		public TFTensor [] Run (TFOutput [] inputs, TFTensor [] inputValues, TFOutput [] outputs, TFOperation [] targetOpers = null, TFBuffer runMetadata = null, TFBuffer runOptions = null, TFStatus status = null)
		{
			CheckDisposed ();

			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (inputValues == null)
				throw new ArgumentNullException (nameof (inputValues));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			var iLen = inputs.Length;
			if (iLen != inputValues.Length)
				throw new ArgumentException ("inputs and inputValues have different lengths", nameof (inputs));
			var oLen = outputs.Length;

			// runOptions and runMetadata might be null
			var cstatus = TFStatus.Setup (status);

			// Create arrays for the unmanaged versions
			var ivals = new IntPtr [iLen];
			for (var i = 0; i < iLen; i++)
				ivals [i] = inputValues [i].Handle;

			// I believe this might not be necessary, the output values in TF_SessionRun looks like a write-only result
			var ovals = new IntPtr [outputs.Length];
			IntPtr [] topers = null;
			var tLen = 0;
			if (targetOpers != null) {
				tLen = targetOpers.Length;
				topers = new IntPtr [tLen];
				for (var i = 0; i < tLen; i++)
					topers [i] = targetOpers [i].Handle;
			}

			unsafe
			{
				TF_SessionRun (Handle, runOptions == null ? null : runOptions.LLBuffer, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, runMetadata == null ? null : runMetadata.LLBuffer, cstatus.Handle);
			}

			cstatus.CheckMaybeRaise (status);
			var result = new TFTensor [oLen];
			for (var i = 0; i < oLen; i++) {
				result [i] = new TFTensor (ovals [i]);
			}
			return result;
		}

		public class PartialRunToken : IDisposable
		{
			internal IntPtr token;

			public void Dispose ()
			{
				if (token == IntPtr.Zero) {
					TF_DeletePRunHandle (token);
					token = IntPtr.Zero;
				}
			}
		}

		public PartialRunToken PartialRunSetup (TFOutput [] inputs, TFOutput [] outputs, TFOperation [] targetOpers, TFStatus status = null)
		{
			CheckDisposed ();

			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			if (targetOpers == null)
				throw new ArgumentNullException (nameof (targetOpers));

			IntPtr returnHandle;
			var cstatus = TFStatus.Setup (status);
			var tLen = targetOpers.Length;
			var topers = new IntPtr [tLen];
			for (var i = 0; i < tLen; i++)
				topers [i] = targetOpers [i].Handle;

			TF_SessionPRunSetup (Handle, inputs, inputs.Length, outputs, outputs.Length, topers, tLen, out returnHandle, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			return new PartialRunToken { token = returnHandle };
		}


		public TFTensor [] PartialRun (PartialRunToken token, TFOutput [] inputs, TFTensor [] inputValues, TFOutput [] outputs, TFOperation [] targetOpers, TFStatus status = null)
		{
			CheckDisposed ();

			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (inputValues == null)
				throw new ArgumentNullException (nameof (inputValues));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			if (targetOpers == null)
				throw new ArgumentNullException (nameof (targetOpers));
			var iLen = inputs.Length;
			if (iLen != inputValues.Length)
				throw new ArgumentException ("inputs and inputValues have different lengths", nameof(inputs));
			
			var oLen = outputs.Length;

			// runOptions and runMetadata might be null
			var cstatus = TFStatus.Setup (status);

			// Create arrays for the unmanaged versions
			var ivals = new IntPtr [iLen];
			for (var i = 0; i < iLen; i++)
				ivals [i] = inputValues [i].Handle;
			var ovals = new IntPtr [oLen];
			var tLen = targetOpers.Length;
			var topers = new IntPtr [tLen];
			for (var i = 0; i < tLen; i++)
				topers [i] = targetOpers [i].Handle;

			TF_SessionPRun (Handle, token.token, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, cstatus.Handle);

			cstatus.CheckMaybeRaise (status);

			var result = new TFTensor [oLen];
			for (var i = 0; i < oLen; i++) {
				result [i] = new TFTensor (ovals [i]);
			}
			return result;
		}
	}
}
