﻿//
// TensorFlow.cs; Bindings to the TensorFlow C API for .NET
// 
// Authors:
//   Miguel de Icaza (miguel@microsoft.com)
//
// Strongly typed API
// The API generally takes a TF_Status that defaults to null, if the value is null, on error, this raises an exception, otherwise, the error is returned on the TF_Status.
// You can use TFStatus.Default for a value to use when you do not want to create the value yourself and are ok reusing the value.
//
// Guidaance on doing language bindings for Tensorflow:
// https://www.tensorflow.org/versions/r0.11/how_tos/language_bindings/
//
//
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Globalization;
using System.Linq;

// We use this TF_Xxx as the native "TF_Xxx *" as those are opaque
using TF_Status = System.IntPtr;
using TF_SessionOptions = System.IntPtr;
using TF_Graph = System.IntPtr;
using TF_OperationDescription = System.IntPtr;
using TF_Operation = System.IntPtr;
using TF_Session = System.IntPtr;
using TF_DeprecatedSession = System.IntPtr;
using TF_Tensor = System.IntPtr;
using TF_ImportGraphDefOptions = System.IntPtr;
using TF_Library = System.IntPtr;
using TF_BufferPtr = System.IntPtr;

using size_t = System.UIntPtr;
using System.Numerics;
using System.Collections.Generic;
using System.Linq.Expressions;

namespace TensorFlow
{
	static partial class NativeBinding
	{
		public const string TensorFlowLibrary = "libtensorflow";

		internal static string GetStr (this IntPtr x) => Marshal.PtrToStringAnsi (x);


	}

	/// <summary>
	/// Contains TensorFlow fundamental methods and utility functions.
	/// </summary>
	public static class TFCore {
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_Version ();

		static TFCore ()
		{
			CheckSize ();
		}

		/// <summary>
		/// Returns the version of the TensorFlow runtime in use.
		/// </summary>
		/// <value>The version.</value>
		public static string Version => TF_Version ().GetStr ();

		// extern size_t TF_DataTypeSize (TF_DataType dt);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern IntPtr TF_DataTypeSize (TFDataType dt);

		/// <summary>
		/// Gets the size in bytes of the specified TensorFlow data type.
		/// </summary>
		/// <returns>The data type size.</returns>
		/// <param name="dt">Dt.</param>
		public static long GetDataTypeSize (TFDataType dt) => (long)TF_DataTypeSize (dt);

		// extern TF_Buffer * TF_GetAllOpList ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_GetAllOpList ();

		/// <summary>
		/// Retrieves the ProtocolBuffer describing all of the available operations in
		/// the TensorFlow library in current use.
		/// </summary>
		/// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
		public static TFBuffer GetAllOpList ()
		{
			return new TFBuffer (TF_GetAllOpList ());
		}

		internal static void CheckSize ()
		{
			unsafe
			{
				if (sizeof (IntPtr) == 4) {
					Console.Error.WriteLine (
						"The TensorFlow native libraries were compiled in 64 bit mode, you must run in 64 bit mode\n" +
						"With Mono, do that with mono --arch=64 executable.exe, if using an IDE like MonoDevelop,\n" +
						"Xamarin Studio or Visual Studio for Mac, Build/Compiler settings, make sure that " +
						"\"Platform Target\" has x64 selected.");
					throw new Exception ();

				}
			}
		}
	}

	/// <summary>
	/// Base class for many TensorFlow data types that provides a common idiom to dispose and
	/// release resources associated with the native data types.   Generally, you do not need to use this.
	/// </summary>
	/// <remarks>
	/// <para>
	/// This implements the Dispose pattern in a reusable form for TensorFlow types.
	/// </para>
	/// <para>
	/// Subclasses invoke the constructor with the handle that this will wrap, and must
	/// override the NativeDispose method (internal) to release the associated resource.
	/// </para>
	/// </remarks>
	public abstract class TFDisposable : IDisposable
	{
		internal IntPtr handle;

		/// <summary>
		/// Returns the opaque handle to the object that this TFDisposable owns.
		/// </summary>
		/// <value>The handle.</value>
		public IntPtr Handle => handle;

		static TFDisposable ()
		{
			TFCore.CheckSize ();
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposable"/> class.
		/// </summary>
		public TFDisposable ()
		{ }

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFDisposable"/> class
		/// from the handle that it will wrap.   
		/// </summary>
		public TFDisposable (IntPtr handle)
		{
			this.handle = handle;
		}

		/// <summary>
		/// Releases all resource used by the <see cref="T:TensorFlow.TFDisposable"/> object.
		/// </summary>
		/// <remarks>Call Dispose when you are finished using the <see cref="T:TensorFlow.TFDisposable"/>. The
		/// Dispose method leaves the <see cref="T:TensorFlow.TFDisposable"/> in an unusable state. After
		/// calling Dispose, you must release all references to the <see cref="T:TensorFlow.TFDisposable"/> so
		/// the garbage collector can reclaim the memory that the <see cref="T:TensorFlow.TFDisposable"/> was occupying.</remarks>
		public void Dispose ()
		{
			Dispose (true);
			GC.SuppressFinalize (this);
		}

		~TFDisposable ()
		{
			Dispose (false);
		}

		// Must be implemented in subclasses to dispose the unmanaged object, it does
		// not need to take care of zeroing out the handle, that is done by the Dispose
		// method inherited from TFDisposable
		internal abstract void NativeDispose (IntPtr handle);

		/// <summary>
		/// Dispose the specified object
		/// </summary>
		/// <param name="disposing">If set to <c>true</c> it means that this method was called from Dispose, otherwise from the finalizer.</param>
		public virtual void Dispose (bool disposing)
		{
			if (disposing) {
				if (handle != IntPtr.Zero)
					NativeDispose (handle);
				handle = IntPtr.Zero;
			}
		}

		internal static void ObjectDisposedException ()
		{
			throw new ObjectDisposedException ("The object was disposed");
		}
	}

	/// <summary>
	/// TensorFlow Exception
	/// </summary>
	public class TFException : Exception {
		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFException"/> class with a message.
		/// </summary>
		/// <param name="message">Message.</param>
		public TFException (string message) : base (message) { }
	}

	/// <summary>
	/// Used to track the result of TensorFlow operations.
	/// </summary>
	/// <remarks>
	/// <para>
	/// TFStatus is used to track the status of a call to some TensorFlow
	/// operations.   Instances of this object are passed to various
	/// TensorFlow operations and you can use the <see cref="P:TensorFlow.TFStatus.Ok"/>
	/// to quickly check if the operation succeeded, or get more detail from the
	/// <see cref="P:TensorFlow.TFStatus.StatusCode"/> and a human-readable text
	/// using the <see cref="P:TensorFlow.TFStatus.StatusMessage"/> property.
	/// </para>
	/// <para>
	/// The convenience <see cref="M:TensorFlow.TFStatus.Raise"/> can be used
	/// to raise a <see cref="P:TensorFlow.TFException"/> if the status of the
	/// operation did not succeed.
	/// </para>
	/// </remarks>
	public class TFStatus : TFDisposable
	{
		// extern TF_Status * TF_NewStatus ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe TF_Status TF_NewStatus ();

		/// <summary>
		/// Per-thread global status that you can use if you do not need to create a new instance of this object.
		/// </summary>
		/// <remarks>
		/// This is provided as a convenience for APIs that take a TFStatus.   While the TFStatus is usually an
		/// optional parameter, when it is made optional, API calls that fail raise an exception.   Use this 
		/// property to pass a TFStatus without having to allocate a new one.   The problem with this of course
		/// is that you risk having multiple parts of your code override this thread-global variable.
		/// </remarks>
		[ThreadStatic] public static TFStatus Default = new TFStatus ();

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFStatus"/> class.
		/// </summary>
		public TFStatus () : base (TF_NewStatus ())
		{
		}

		// extern void TF_DeleteStatus (TF_Status *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe void TF_DeleteStatus (TF_Status status);

		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteStatus (handle);
		}


		// extern void TF_SetStatus (TF_Status *s, TF_Code code, const char *msg);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetStatus (TF_Status s, TFCode code, string msg);

		/// <summary>
		/// Sets the status code on this TFStatus.
		/// </summary>
		/// <param name="code">Code.</param>
		/// <param name="msg">Message.</param>
		public void SetStatusCode (TFCode code, string msg)
		{
			TF_SetStatus (handle, code, msg);
		}

		// extern TF_Code TF_GetCode (const TF_Status *s);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe TFCode TF_GetCode (TF_Status s);

		/// <summary>
		/// Gets the status code for the status code.
		/// </summary>
		/// <value>The status code as an enumeration.</value>
		public TFCode StatusCode {
			get {
				return TF_GetCode (handle);
			}
		}

		// extern const char * TF_Message (const TF_Status *s);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_Message (TF_Status s);

		/// <summary>
		/// Gets a human-readable status message.
		/// </summary>
		/// <value>The status message.</value>
		public string StatusMessage => TF_Message (handle).GetStr ();

		/// <summary>
		/// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.
		/// </summary>
		/// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFStatus"/>.</returns>
		public override string ToString ()
		{
			return string.Format ("[TFStatus: StatusCode={0}, StatusMessage={1}]", StatusCode, StatusMessage);
		}


		/// <summary>
		/// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to ok.
		/// </summary>
		/// <value><c>true</c> if ok; otherwise, <c>false</c>.</value>
		public bool Ok => StatusCode == TFCode.Ok;

		/// <summary>
		/// Gets a value indicating whether this <see cref="T:TensorFlow.TFStatus"/> state has been set to an error.
		/// </summary>
		/// <value><c>true</c> if error; otherwise, <c>false</c>.</value>
		public bool Error => StatusCode != TFCode.Ok;

		/// <summary>
		/// Convenience method that raises an exception if the current status is an error.
		/// </summary>
		/// <remarks>
		/// You can use this method as a convenience to raise an exception after you
		/// invoke an operation if the operation did not succeed.
		/// </remarks>
		public void Raise ()
		{
			if (TF_GetCode (handle) != TFCode.Ok)
				throw new TFException (StatusMessage);
		}

		// 
		// Utility function used to simplify implementing the idiom
		// where the user optionally provides a TFStatus, if it is provided,
		// the error is returned there;   If it is not provided, then an
		// exception is raised.
		//

		internal bool CheckMaybeRaise (TFStatus incomingStatus, bool last = true)
		{
			if (incomingStatus == null) {
				if (handle == IntPtr.Zero)
					Console.WriteLine ("oops");
				if (StatusCode != TFCode.Ok) {
					var e = new TFException (StatusMessage);
					Dispose ();
					throw e;
				}
				if (last)
					Dispose ();
				return true;
			}
			return StatusCode == TFCode.Ok;
		}

		internal static TFStatus Setup (TFStatus incoming)
		{
			return incoming == null ? new TFStatus () : incoming;
		}
	}

	internal class TFString
	{
		// extern size_t TF_StringEncode (const char *src, size_t src_len, char *dst, size_t dst_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe size_t TF_StringEncode (byte* src, size_t src_len, sbyte* dst, size_t dst_len, TF_Status status);
		
		// extern size_t TF_StringDecode (const char *src, size_t src_len, const char **dst, size_t *dst_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe size_t TF_StringDecode (sbyte* src, size_t src_len, sbyte** dst, size_t* dst_len, TF_Status status);

		// extern size_t TF_StringEncodedSize (size_t len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern size_t TF_StringEncodedSize (size_t len);
	}

	/// <summary>
	/// The session options object holds configuration options that you want to use during your session, like the TensorFlow target or the configuration.
	/// </summary>
	public class TFSessionOptions : TFDisposable
	{
		// extern TF_SessionOptions * TF_NewSessionOptions ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe TF_SessionOptions TF_NewSessionOptions ();

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFSessionOptions"/> class.
		/// </summary>
		public TFSessionOptions () : base (TF_NewSessionOptions ()) { }

		// extern void TF_DeleteSessionOptions (TF_SessionOptions *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe void TF_DeleteSessionOptions (TF_SessionOptions options);
		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteSessionOptions (handle);
		}

		// extern void TF_SetTarget (TF_SessionOptions *options, const char *target);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetTarget (TF_SessionOptions options, string target);

		/// <summary>
		/// Sets the target in options.
		/// </summary>
		/// <param name="target">target can be empty, a single entry, or a comma separated list of entries.
		/// Each entry is in one of the following formats: "local", ip:port, host:port.</param>
		/// 
		public void SetTarget (string target)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			
			TF_SetTarget (handle, target);
		}

		// extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetConfig (TF_SessionOptions options, IntPtr proto, size_t proto_len, TF_Status status);

		/// <summary>
		/// Sets the configuration information for the session.
		/// </summary>
		/// <param name="protoData">Serialized protocol buffer for the tensorflow.ConfigProto message.</param>
		/// <param name="length">Length of the buffer.</param>
		/// <param name="status">If config was not parsed successfully as a ConfigProto, the error is recorded here.</param>
		/// <remarks>
		/// The configuration option is a Protocol Buffer representing the tensorflow.ConfigProto
		/// </remarks>
		public void SetConfig (IntPtr protoData, int length, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();

			var cstatus = TFStatus.Setup (status);

			TF_SetConfig (handle, protoData, (UIntPtr)length, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
		}

	}

	/// <summary>
	/// Represents a computation graph.  Graphs may be shared between sessions and are thread safe.
	/// </summary>
	/// <remarks>
	/// <para>
	/// Graphs consist of operations (represented by TFOperation objects), these can be named, or 
	/// the runtime will automatically assign a name.
	/// </para> 
	/// <para>
	/// For debugging purposes, you might want to group operations together, for this, call the
	/// WithScope method with your new scope, which will create a new namespace for your object names.
	/// </para>
	/// <para>
	/// For example, if you call WithScope ("demo"), and add an operation named "add" inside the
	/// scope, the full name of the operation will be "demo/add", if you create a new scope inside, say
	/// "hot", and add a "sub" operation there the result will be "demo/hot/sub".
	/// </para>
	/// </remarks>
	public partial class TFGraph : TFDisposable
	{
		// extern TF_Graph * TF_NewGraph ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Graph TF_NewGraph ();

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFGraph"/> class.
		/// </summary>
		public TFGraph () : base (TF_NewGraph ())
		{
		}

		internal TFGraph (IntPtr handle) : base (handle)
		{
		}

		// extern void TF_DeleteGraph (TF_Graph *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteGraph (TF_Graph graph);
		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteGraph (handle);
		}

		// extern void TF_GraphSetTensorShape (TF_Graph *graph, TF_Output output, const int64_t *dims, const int num_dims, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphSetTensorShape (TF_Graph graph, TFOutput output, long [] dims, int num_dims, TF_Status status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphSetTensorShape (TF_Graph graph, TFOutput output, IntPtr dims, int num_dims, TF_Status status);

		/// <summary>
		/// Sets the tensor shape of the tensor referenced by <paramref name="output"/> to the shape described by <paramref name="dims"/>.
		/// </summary>
		/// <param name="output">The tensor on which this method will operate in the graph.</param>
		/// <param name="dims">The tensor shape, specified as an array of dimensions.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public void SetTensorShape (TFOutput output, long [] dims, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();

			var cstatus = TFStatus.Setup (status);
			if (dims == null)
				TF_GraphSetTensorShape (handle, output, IntPtr.Zero, 0, cstatus.handle);
			else
				TF_GraphSetTensorShape (handle, output, dims, dims.Length, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
		}

		// extern int TF_GraphGetTensorNumDims (TF_Graph *graph, TF_Output output, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_GraphGetTensorNumDims (TF_Graph graph, TFOutput output, TF_Status status);

		public int GetTensorNumDims (TFOutput output, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var code = TF_GraphGetTensorNumDims (handle, output, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return code;
		}

		// extern void TF_GraphGetTensorShape (TF_Graph *graph, TF_Output output, int64_t *dims, int num_dims, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphGetTensorShape (TF_Graph graph, TFOutput output, long [] dims, int num_dims, TF_Status status);

		/// <summary>
		/// Returns the shape of a tensor specified in <paramref name="output"/>.
		/// </summary>
		///
		/// <returns>The tensor shape.    If the number of dimensions in the shape is unknown or the shape is, a scalar, the values in the array will be zero. Otherwise, each element of will be set corresponding to the size of the dimension. An  unknown dimension is represented by -1.</returns>
		/// <param name="output">The tensor that you want to look up.  </param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public long [] GetTensorShape (TFOutput output, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var n = TF_GraphGetTensorNumDims (handle, output, cstatus.handle);
			if (!cstatus.CheckMaybeRaise (status, last: false))
				return null;
			
			var dims = new long [n];
			TF_GraphGetTensorShape (handle, output, dims, dims.Length, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return dims;
		}

		// extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphToGraphDef (TF_Graph graph, LLBuffer* output_graph_def, TF_Status status);

		/// <summary>
		/// Write out a serialized representation of the graph (as a GraphDef protocol buffer message) into <paramref name="outputGraphDef"/>.
		/// </summary>
		/// <param name="outputGraphDef">Target buffer where the graphs is serialized into.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public void ToGraphDef (TFBuffer outputGraphDef, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (outputGraphDef == null)
				throw new ArgumentNullException (nameof (outputGraphDef));

			var cstatus = TFStatus.Setup (status);
			unsafe
			{
				TF_GraphToGraphDef (handle, outputGraphDef.LLBuffer, cstatus.handle);
			}
			cstatus.CheckMaybeRaise (status);
		}

		// extern void TF_GraphImportGraphDef (TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphImportGraphDef (TF_Graph graph, LLBuffer* graph_def, TF_ImportGraphDefOptions options, TF_Status status);

		/// <summary>
		/// Import a serialized graph into this graph, using the specified prefix.
		/// </summary>
		/// <returns>The import.</returns>
		/// <param name="graphDef">A buffer containing the serialized graph.</param>
		/// <param name="prefix">A prefix that will be prepended to names of nodes in the <paramref name="graphDef"/> when they are imported into the graph.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public void Import (TFBuffer graphDef, string prefix = "", TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (graphDef == null)
				throw new ArgumentNullException (nameof (graphDef));
			if (prefix == null)
				throw new ArgumentNullException (nameof (prefix));

			using (var options = new TFImportGraphDefOptions ()) {
				options.SetPrefix (prefix);
				Import (graphDef, options, status);
			}
		}

		/// <summary>
		/// Import a serialized graph into this graph, using the specified importing options.
		/// </summary>
		/// <returns>The import.</returns>
		/// <param name="graphDef">A buffer containing the serialized graph.</param>
		/// <param name="options">Importing graph options.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public void Import (TFBuffer graphDef, TFImportGraphDefOptions options, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (graphDef == null)
				throw new ArgumentNullException (nameof (graphDef));
			if (options == null)
				throw new ArgumentNullException (nameof (options));

			var cstatus = TFStatus.Setup (status);
			unsafe
			{
				TF_GraphImportGraphDef (handle, graphDef.LLBuffer, options.handle, cstatus.handle);
			}
			cstatus.CheckMaybeRaise (status);
		}

		/// <summary>
		/// Import a serialized graph held in a byte array into this graph, using the specified prefix.
		/// </summary>
		/// <returns>The import.</returns>
		/// <param name="buffer">A byte array containing the serialized graph.</param>
		/// <param name="prefix">A prefix that will be prepended to names of nodes in the graph when they are imported into the graph.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>		
		public void Import (byte [] buffer, string prefix = "", TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));
			if (prefix == null)
				throw new ArgumentNullException (nameof (prefix));
			using (var options = new TFImportGraphDefOptions ()) {
				options.SetPrefix (prefix);
				Import (buffer, options, status);
			}
		}

		/// <summary>
		/// Import a serialized graph held in a byte array into this graph, using the specified import options.
		/// </summary>
		/// <returns>The import.</returns>
		/// <param name="buffer">A byte array containing the serialized graph.</param>
		/// <param name="options">Importing graph options.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public void Import (byte [] buffer, TFImportGraphDefOptions options, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));
			if (options == null)
				throw new ArgumentNullException (nameof (options));
			var cstatus = TFStatus.Setup (status);
			using (var tb = new TFBuffer (buffer, 0, buffer.Length)) 
				Import (tb, options, status);
			
			cstatus.CheckMaybeRaise (cstatus);
		}

		// extern TF_Operation * TF_GraphOperationByName (TF_Graph *graph, const char *oper_name);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_GraphOperationByName (TF_Graph graph, string oper_name);

		/// <summary>
		/// Gets the <see cref="T:TensorFlow.TFGraph"/> with the specified name, or null if the named operation does not exist in the graph.
		/// </summary>
		/// <param name="name">Name to lookup.</param>
		public TFOperation this [string name] {
			get {
				if (handle == IntPtr.Zero)
					ObjectDisposedException ();
				var h = TF_GraphOperationByName (handle, name);
				if (h == IntPtr.Zero)
					return null;
				return new TFOperation (this, h);
			}
		}

		// extern TF_Operation * TF_GraphNextOperation (TF_Graph *graph, size_t *pos);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_GraphNextOperation (TF_Graph graph, ref IntPtr token);

		/// <summary>
		/// Returns the enumerator that returns all the TFOperations in a graph.
		/// </summary>
		/// <returns>The enumerator.</returns>
		public IEnumerable<TFOperation> GetEnumerator ()
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			IntPtr token = IntPtr.Zero;
			IntPtr operll;

			while ((operll = TF_GraphNextOperation (handle, ref token)) != IntPtr.Zero)
				yield return new TFOperation (this, operll);
		}

		/// <summary>
		///  Returns the tensor shape for the specific output pparameters as an array of longs.
		/// </summary>
		/// <returns>null for single dimension, .</returns>
		/// <param name="output">The output operation to probe.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public long [] GetShape (TFOutput output, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var ndims = TF_GraphGetTensorNumDims (handle, output, cstatus.handle);
			if (!cstatus.CheckMaybeRaise (status, last: false))
				return null;
			
			if (ndims == 0)
				return null;
			var ret = new long [ndims];
			TF_GraphGetTensorShape (handle, output, ret, ndims, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return ret;
		}

		/// <summary>
		/// Returns the current name scope in use, to change this, use the WithScope method.
		/// </summary>
		/// <value>The current name scope.</value>
		public string CurrentNameScope { get; internal set; } = "";

		/// <summary>
		/// Creates a new namescope by setting the scope to the description provided.
		/// </summary>
		/// <returns>A new scope that will remain in use until the return TFScope is disposed.</returns>
		/// <param name="nameScopeDesc">The namescope description, if the value is null, this
		/// will reset the toplevel namescope to be the empty value. </param>
		/// <remarks>
		/// <para>
		/// To more easily name your operations and group then, you can use the
		/// WithScope method to set a current name scope that alter the complete name
		/// of an operation added to the graph.
		/// </para>
		/// <para>
		/// The graph starts with a scope set to the empty string, you can introduce new
		/// scopes by calling WithScope, and can be conveniently used with the C# using
		/// statement, like this:
		/// </para>
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

		Dictionary<string, int> values = new Dictionary<string, int> ();

		string MakeName (string operName, string userName)
		{
			if (userName == null) {
				var k = CurrentNameScope == "" ? operName : CurrentNameScope + "/" + operName;

				return MakeUnique (k);
			}
			if (CurrentNameScope == "")
				return userName;
			return CurrentNameScope + "/" + userName;
		}

		string MakeUnique (string name)
		{
			int val = 0;

			if (!values.TryGetValue (name, out val))
				val = 0;
			else
				val++;
			values [name] = val;
			return name + val;
		}

		internal int LastId;
		internal int GetNextId ()
		{
			return LastId++;
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		unsafe extern static void TF_GraphImportGraphDefWithReturnOutputs (
			TF_Graph graph, LLBuffer *graph_def,
			TF_ImportGraphDefOptions options, TFOutput *return_outputs,
			int num_return_outputs, TF_Status status);

		/// <summary>
		/// Imports a graph serialized into the graph
		/// </summary>
		/// <param name="graphDef">Serialized graph definition (in protocol buffer format).</param>
		/// <param name="options">Import options.</param>
		/// <param name="returnOutputs">Array large enough to contain all the return options.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public void ImportGraphDef (TFBuffer graphDef, TFImportGraphDefOptions options, TFOutput [] returnOutputs, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (graphDef == null)
				throw new ArgumentNullException (nameof (graphDef));
			if (options == null)
				throw new ArgumentNullException (nameof (options));
			var cstatus = TFStatus.Setup (status);

			unsafe
			{
				if (returnOutputs == null) {
					TF_GraphImportGraphDefWithReturnOutputs (handle, graphDef.LLBuffer, options.handle, null, 0, cstatus.handle);
				} else {
					fixed (TFOutput* first = &returnOutputs [0])
					{
						TF_GraphImportGraphDefWithReturnOutputs (handle, graphDef.LLBuffer, options.handle, first, returnOutputs.Length, cstatus.handle);
					}
				}
			}
		}

		[StructLayout (LayoutKind.Sequential)]
		unsafe struct TFWhileParams
		{
			public int ninputs;
			public TF_Graph cond_graph;
			public TFOutput* cond_inputs;
			public TFOutput cond_output;
			public TF_Graph body_graph;
			public TFOutput* body_inputs;
			public TFOutput* body_outputs;
			public IntPtr charPtrName;
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TFWhileParams TF_NewWhile (TF_Graph g, TFOutput [] inputs, int ninputs, TF_Status status);

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern void TF_AbortWhile (ref TFWhileParams pars);

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_FinishWhile (ref TFWhileParams pars, TF_Status status, TFOutput *outputs);

		static unsafe TFOutput [] CopyFrom (TFOutput* ptr, int n)
		{
			var r = new TFOutput [n];
			for (int i = 0; i < n; i++)
				r [i] = ptr [i];

			return r;
		}

		/// <summary>
		/// Signature of the method that will be invoked by the TFGraph.While method to construct a while loop
		/// </summary>
		/// <remarks>
		/// <para>
		/// The method should build up the condition on the conditionGraph and the body of the while 
		/// loop in the provided bodyGraph.   It should set the condOutput to the value used as the
		/// condition output and the array of values in bodyOutputs to the final outputs as well as the
		/// name to be used, if not set, one will be assigned.
		/// </para>
		/// <para>
		/// The conditionGraph represents the while condition and the inputs are the current values of the
		/// input variables (condInputs).   The output should be a scalar boolean.
		/// </para>
		/// <para>
		/// The loop body graph is in bodyGraph, The inputs are the current values of the loop
		/// variables. The outputs are the updated values of the loop variables.
		/// </para>
		/// <para>
		/// You can use the passed status record problems with it.
		/// </para>
		/// </remarks>
		public delegate void WhileConstructor (TFGraph conditionGraph, TFOutput [] condInputs, out TFOutput condOutput, TFGraph bodyGraph, TFOutput [] bodyInputs, TFOutput [] bodyOutputs, out string name);

		/// <summary>
		/// Constructs a while loop with the specified inputs and a callback that composes the while loop
		/// </summary>
		/// <param name="inputs">Inputs.</param>
		/// <param name="constructor">Callback method that fills out the various while loop parameters.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		/// <returns>
		/// An array of TFOutputs from creating the While loop, or null if there is an error creating the 
		/// while loop, or if the constructor raised an exception when it was invoked.
		/// </returns>
		public TFOutput [] While (TFOutput [] inputs, WhileConstructor constructor, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (constructor == null)
				throw new ArgumentNullException (nameof (constructor));
			var cstatus = TFStatus.Setup (status);
			TFWhileParams result = TF_NewWhile (handle, inputs, inputs.Length, cstatus.handle);
			if (cstatus.Error)
				return null;
			
			try {

				// 
				// Call constructor here
				// Wrap the various TF_graphs (with owns=false)
				// Marshal the condInputs, bodyInputs
				//
				string name;

				int n = result.ninputs;
				TFOutput [] bodyOutputs = new TFOutput [n];
				unsafe
				{
					var condGraph = new TFGraphUnowned (result.cond_graph);
					var bodyGraph = new TFGraphUnowned (result.body_graph);
					constructor (condGraph, CopyFrom (result.cond_inputs, n), out result.cond_output, bodyGraph, CopyFrom (result.body_inputs, n), bodyOutputs, out name);
				}
				if (name == null || name == "")
					name = MakeUnique ("while");
				// On return, copy the condOutput and bodyOututs
				var text = Encoding.UTF8.GetBytes (name);

				result.charPtrName = Marshal.AllocHGlobal (text.Length + 1);
				Marshal.Copy (text, 0, result.charPtrName, text.Length);
				Marshal.WriteByte (result.charPtrName, text.Length, 0);

				unsafe
				{
					for (int i = 0; i < n; i++)
						result.body_outputs [i] = bodyOutputs [i];
					var ret = new TFOutput [inputs.Length];
					fixed (TFOutput* first = &ret [0])
						TF_FinishWhile (ref result, cstatus.handle, first);


					if (cstatus.CheckMaybeRaise (status))
						return ret;
				}
				return null;
			} catch {
				TF_AbortWhile (ref result);
				return null;
			}
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddGradients (TF_Graph graph, TFOutput* ys, int ny, TFOutput* xs, int nx, TFOutput* dx, TF_Status status, TFOutput* dy);

		/// <summary>
		/// Adds a gradient: the operations needed to compute the partial derivatives of sum of <paramref name="y"/>` wrt to <paramref name="x"/>.
		/// </summary>
		/// <returns>The partial derivatives, the size of the array is the same as the length of the <paramref name="y"/> array.</returns>
		/// <param name="y">The y elements.</param>
		/// <param name="x">The x elements.</param>
		/// <param name="dx">Initial gradients, which represent the symbolic partial derivatives of some loss function `L` w.r.t. <paramref name="y"/> ).   
		/// If the parameter is null, the implementation will use dx for 'OnesLike' for all shapes in <paramref name="y"/></param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		/// <remarks>
		/// d(y[0] + y[1]+ ...)/dx[0], d(y[0] + y[1] + ...)/dx[1]z...
		/// </remarks>
		public TFOutput [] AddGradients (TFOutput [] y, TFOutput [] x, TFOutput [] dx = null, TFStatus status = null)
		{
			if (y == null)
				throw new ArgumentNullException (nameof (y));
			if (x == null)
				throw new ArgumentNullException (nameof (x));
			if (dx != null) {
				if (dx.Length != y.Length)
					throw new ArgumentException ("If dx is not null, the size of the gradients must match the size of y", nameof (dx));
			}

			var cstatus = TFStatus.Setup (status);

			var ret = new TFOutput [x.Length];
			unsafe
			{
				fixed (TFOutput* pret = &ret [0]) {
					fixed (TFOutput* py = &y [0]) {
						fixed (TFOutput* px = &x [0]) {
							if (dx == null) {
								TF_AddGradients (handle, py, y.Length, px, x.Length, (TFOutput*)null, status.Handle, pret);
							} else {
								fixed (TFOutput* pdx = &dx [0]) {
									TF_AddGradients (handle, py, y.Length, px, x.Length, pdx, status.Handle, pret);
								}
							}
						}
					}
				}
			}
			if (!cstatus.CheckMaybeRaise (status, last: false))
				return null;
			return ret;
		}
	}

	//
	// A TFGraph that will not release the undelying handle, this is used
	// when we want to surface a TFGraph that we do not own, so we do not
	// want to delete the handle when this object is collected
	//
	internal class TFGraphUnowned : TFGraph
	{
		internal TFGraphUnowned (IntPtr handle) : base (handle)
		{
		}

		internal override void NativeDispose (TF_Status handle)
		{
			// nothing, we do not own the handle
		}
	}

	/// <summary>
	/// TFGraph name scope handle
	/// </summary>
	/// <remarks>
	/// Instances of this class when disposed restore the CurrentNameScope to the
	/// value they had when the TFGraph.WithScope method was called.
	/// </remarks>
	public class TFScope : IDisposable 
	{
		TFGraph container;
		string name;

		internal TFScope (TFGraph container)
		{
			this.container = container;
			name = container.CurrentNameScope;
		}

		/// <summary>
		/// Pops the name space to the previous namescope in use.
		/// </summary>
		/// <remarks>Call <see cref="Dispose"/> when you are finished using the <see cref="T:TensorFlow.TFScope"/>
		/// to restore the previous name scope in use in the <see cref="T:TensorFlow.TFGraph"/>.
		/// </remarks>
		public void Dispose ()
		{
			container.CurrentNameScope = name;
		}
	}

	/// <summary>
	/// Low-level TensorFlow operation builder
	/// </summary>
	/// <remarks>
	/// <para>This is the low-level API that is used to create operations by manually specificying all
	/// the parameters of an operation (inputs, outputs, attribute descriptions) that can then
	/// be attached into a graph.
	/// </para>
	/// <para>
	/// Generally, you will instead be using the methods surfaced in <see cref="T:TensorFlow.TFGraph"/> 
	/// that surfaces a C# high-level API that has already been bound to the built-in TensorFlow
	/// nodes.
	/// </para>
	/// <para>
	/// You create instances bound to a graph, add inputs, attributes and so on, and when you are done
	/// you can call the <see cref="FinishOperation"/> method that will turn this TFOperationDesc 
	/// into a <see cref="T:TensorFlow.TFOperation"/>.
	/// </para>
	/// </remarks>
	public class TFOperationDesc : TFDisposable
	{
		string opType, operName;
		TFGraph graph;

		// extern TF_OperationDescription * TF_NewOperation (TF_Graph *graph, const char *op_type, const char *oper_name);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_OperationDescription TF_NewOperation (TF_Graph graph, string opType, string oper_name);

		public TFOperationDesc (TFGraph graph, string opType, string operName) : base (IntPtr.Zero)
		{
			if (graph == null)
				throw new ArgumentNullException ("graph");

			handle = TF_NewOperation (graph.handle, opType, operName);
			this.graph = graph;
			this.opType = opType;
			this.operName = operName;
		}

		internal override void NativeDispose (IntPtr handle)
		{
			// If you reach this, you never called FinishOperation
			Console.WriteLine ($"TFOperationDescription({opType},{operName} was never turned into an TFOperation");
		}

		// extern void TF_SetDevice (TF_OperationDescription *desc, const char *device);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetDevice (TF_OperationDescription desc, string device);

		public TFOperationDesc SetDevice (string device)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (device == null)
				throw new ArgumentNullException ("device");
			TF_SetDevice (handle, device);
			return this;
		}

		// extern void TF_AddInput (TF_OperationDescription *desc, TF_Output input);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddInput (TF_OperationDescription desc, TFOutput input);

		/// <summary>
		/// Adds the specified input to the operation
		/// </summary>
		/// <returns>The input.</returns>
		/// <param name="input">Input.</param>
		public TFOperationDesc AddInput (TFOutput input)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			TF_AddInput (handle, input);
			return this;
		}

		// extern void TF_AddInputList (TF_OperationDescription *desc, const TF_Output *inputs, int num_inputs);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddInputList (TF_OperationDescription desc, TFOutput [] inputs, int num_inputs);

		/// <summary>
		/// Adds a series of inputs to the operation.
		/// </summary>
		/// <param name="inputs">Inputs, this is a params array for your convenience.</param>
		public TFOperationDesc AddInputs (params TFOutput [] inputs)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (inputs == null || inputs.Length == 0)
				return this;

			TF_AddInputList (handle, inputs, inputs.Length);
			return this;
		}

		// extern void TF_AddControlInput (TF_OperationDescription *desc, TF_Operation *input);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddControlInput (TF_OperationDescription desc, TF_Operation input);

		/// <summary>
		/// Ensure that the operation does not execute before the control operation does.
		/// </summary>
		/// <param name="control">Operation that must be executed before running this operation.</param>
		/// <remarks>
		/// <para>
		/// A control input is an Operation that must be executed before running the operation 
		/// currently being built.  
		/// </para>
		/// <para>
		/// For example, an Assert operation may be added as a control input for this operation. 
		/// The Assert now behaves as a pre-condition that will always verify itself before
		/// running the operation.
		/// </para>
		/// </remarks>
		public TFOperationDesc AddControlInput (TFOperation control)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (control == null)
				throw new ArgumentNullException ("input");

			TF_AddControlInput (handle, control.handle);
			return this;
		}

		// extern void TF_ColocateWith (TF_OperationDescription *desc, TF_Operation *op);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_ColocateWith (TF_OperationDescription desc, TF_Operation op);

		public TFOperationDesc ColocateWith (TFOperation op)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (op == null)
				throw new ArgumentNullException ("op");
			TF_ColocateWith (handle, op.handle);
			return this;
		}

		// extern void TF_SetAttrString (TF_OperationDescription *desc, const char *attr_name, const void *value, size_t length);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrString (TF_OperationDescription desc, string attr_name, IntPtr value, size_t length);

		public TFOperationDesc SetAttr (string attrName, string value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			var bytes = Encoding.UTF8.GetBytes (value);
			var buf = Marshal.AllocHGlobal (bytes.Length + 1);
			Marshal.Copy (bytes, 0, buf, bytes.Length);

			TF_SetAttrString (handle, attrName, buf, (UIntPtr)bytes.Length);
			return this;
		}

		// extern void TF_SetAttrStringList (TF_OperationDescription *desc, const char *attr_name, const void *const *values, const size_t *lengths, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrStringList (TF_OperationDescription desc, string attr_name, IntPtr [] values, UIntPtr [] lengths, int num_values);
		public TFOperationDesc SetAttr (string attrName, string [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			int n = values.Length;
			var unmanaged = new IntPtr [n];
			var lenghts = new UIntPtr [n];
			for (int i = 0; i < n; i++) {
				var bytes = Encoding.UTF8.GetBytes (values [i]);
				var buf = Marshal.AllocHGlobal (bytes.Length + 1);
				var bc = bytes.Length;

				Marshal.Copy (bytes, 0, buf, bc);
				unmanaged [i] = buf;
				lenghts [i] = (size_t)bc;
			}
			TF_SetAttrStringList (handle, attrName, unmanaged, lenghts, n);
			return this;
		}


		// extern void TF_SetAttrInt (TF_OperationDescription *desc, const char *attr_name, int64_t value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrInt (TF_OperationDescription desc, string attr_name, long value);

		public TFOperationDesc SetAttr (string attrName, long value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrInt (handle, attrName, value);
			return this;
		}

		// extern void TF_SetAttrIntList (TF_OperationDescription *desc, const char *attr_name, const int64_t *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrIntList (TF_OperationDescription desc, string attr_name, long [] values, int num_values);

		public TFOperationDesc SetAttr (string attrName, long [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrIntList (handle, attrName, values, values.Length);
			return this;
		}


		// extern void TF_SetAttrFloat (TF_OperationDescription *desc, const char *attr_name, float value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrFloat (TF_OperationDescription desc, string attr_name, float value);

		public TFOperationDesc SetAttr (string attrName, float value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrFloat (handle, attrName, value);
			return this;
		}

		// extern void TF_SetAttrFloatList (TF_OperationDescription *desc, const char *attr_name, const float *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrFloatList (TF_OperationDescription desc, string attr_name, float [] values, int num_values);

		public TFOperationDesc SetAttr (string attrName, float [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrFloatList (handle, attrName, values, values.Length);
			return this;
		}

		// extern void TF_SetAttrBool (TF_OperationDescription *desc, const char *attr_name, unsigned char value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrBool (TF_OperationDescription desc, string attr_name, byte value);

		public TFOperationDesc SetAttr (string attrName, bool value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrBool (handle, attrName, (byte)(value ? 1 : 0));
			return this;
		}

		// extern void TF_SetAttrBoolList (TF_OperationDescription *desc, const char *attr_name, const unsigned char *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrBoolList (TF_OperationDescription desc, string attr_name, bool [] values, int num_values);

		public TFOperationDesc SetAttr (string attrName, bool [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrBoolList (handle, attrName, values, values.Length);
			return this;
		}

		// extern void TF_SetAttrType (TF_OperationDescription *desc, const char *attr_name, TF_DataType value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrType (TF_OperationDescription desc, string attr_name, TFDataType value);

		public TFOperationDesc SetAttrType (string attrName, TFDataType dataType)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrType (handle, attrName, dataType);
			return this;
		}

		// extern void TF_SetAttrTypeList (TF_OperationDescription *desc, const char *attr_name, const TF_DataType *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTypeList (TF_OperationDescription desc, string attr_name, TFDataType [] values, int num_values);

		public TFOperationDesc SetAttrType (string attrName, params TFDataType [] dataType)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (dataType == null)
				throw new ArgumentNullException (nameof (dataType));
			TF_SetAttrTypeList (handle, attrName, dataType, dataType.Length);
			return this;
		}

		// extern void TF_SetAttrShape (TF_OperationDescription *desc, const char *attr_name, const int64_t *dims, int num_dims);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, long [] dims, int num_dims);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, IntPtr dims, int num_dims);

		public TFOperationDesc SetAttrShape (string attrName, TFShape shape)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (shape == null || shape.dims  == null)
				TF_SetAttrShape (handle, attrName, null, -1);
			else
				TF_SetAttrShape (handle, attrName, shape.dims, shape.dims.Length);
			return this;
		}

		// extern void TF_SetAttrShapeList (TF_OperationDescription *desc, const char *attr_name, const int64_t *const *dims, const int *num_dims, int num_shapes);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShapeList (TF_OperationDescription desc, string attr_name, IntPtr dims, int [] num_dims, int num_shapes);

		public TFOperationDesc SetAttrShape (string attrName, TFShape [] shapeList)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (shapeList == null)
				throw new ArgumentNullException (nameof (shapeList));
			int num_shapes = shapeList.Length;
			var num_dims = new int [shapeList.Length];
			unsafe
			{
				var unmanaged = Marshal.AllocHGlobal (sizeof (IntPtr) * num_shapes);
				int ofs = 0;
				for (int i = 0; i < num_shapes; i++) {
					IntPtr array = Marshal.AllocHGlobal (sizeof (long) * shapeList [i].dims.Length);
					Marshal.Copy (shapeList [i].dims, 0, array, shapeList [i].dims.Length);
					Marshal.WriteIntPtr (unmanaged, ofs, array);
					ofs += sizeof (IntPtr);
				}
				TF_SetAttrShapeList (handle, attrName, unmanaged, num_dims, num_shapes);
				ofs = 0;
				for (int i = 0; i < num_shapes; i++) {
					var ptr = Marshal.ReadIntPtr (unmanaged, ofs);
					Marshal.FreeHGlobal (ptr);
					ofs += sizeof (IntPtr);
				}
				Marshal.FreeHGlobal (unmanaged);
			}
			return this;
		}

		// extern void TF_SetAttrTensorShapeProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorShapeProto (TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status);
		public TFOperationDesc SetAttrTensorShapeProto (string attrName, IntPtr proto, size_t protoLen, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			TF_SetAttrTensorShapeProto (handle, attrName, proto, protoLen, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return this;
		}

		// extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription *desc, const char *attr_name, const void *const *protos, const size_t *proto_lens, int num_shapes, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, void** protos, size_t* proto_lens, int num_shapes, TF_Status status);
		// TODO:

		// extern void TF_SetAttrTensor (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensor (TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status);

		public TFOperationDesc SetAttr (string attrName, TFTensor tensor, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (tensor == null)
				throw new ArgumentNullException ("tensor");
			var cstatus = TFStatus.Setup (status);

			TF_SetAttrTensor (handle, attrName, tensor.handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return this;
		}

		// extern void TF_SetAttrTensorList (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *const *values, int num_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorList (TF_OperationDescription desc, string attr_name, IntPtr [] values, int num_values, TF_Status status);
		public TFOperationDesc SetAttr (string attrName, TFTensor [] tensor, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (tensor == null)
				throw new ArgumentNullException (nameof (tensor));
			var cstatus = TFStatus.Setup (status);
			var unmanaged = new IntPtr [tensor.Length];
			for (int i = 0; i < tensor.Length; i++)
				unmanaged [i] = tensor [i].handle;
			TF_SetAttrTensorList (handle, attrName, unmanaged, unmanaged.Length, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return this;
		}

		// extern void TF_SetAttrValueProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrValueProto (TF_OperationDescription desc, string attr_name, void* proto, size_t proto_len, TF_Status status);
		// TODO:

		// extern TF_Operation * TF_FinishOperation (TF_OperationDescription *desc, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_FinishOperation (TF_OperationDescription desc, TF_Status status);

		/// <summary>
		/// Turns the operation description into an actual operation in the graph.
		/// </summary>
		/// <returns>The operation on success, or null on error.</returns>
		/// <param name="status">Optional status, on failure the operation is not added to the graph.  If you pass null (the default), this operation throws on error conditions.</param>
		public TFOperation FinishOperation (TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var h = TF_FinishOperation (handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			handle = IntPtr.Zero;
			GC.SuppressFinalize (this);
			if (status != null && status.Error)
				return null;

			return new TFOperation (graph, h);
		}
	}

	/// <summary>
	/// Represents a computation node in the graph.  Tensorflow operations are attached to a <see cref="T:Tensorflow.TFGraph"/>.
	/// </summary>
	/// <remarks>
	/// TFOperations are usually created by  invoking one of the methods in 
	/// <see cref="T:Tensorflow.TFGraph"/>, but they can also be constructed
	/// manually using the low-level <see cref="T:Tensorflow.TFOperationDesc"/> API.
	/// </remarks>
	public partial class TFOperation
	{
		internal IntPtr handle;

		/// <summary>
		/// Gets the handle to the unmanaged TF_Operation object.
		/// </summary>
		/// <value>The handle.</value>
		public IntPtr Handle => handle;

		// Pointer to the graph, to keep it from collecting if there are TFOperations alive.
		internal TFGraph graph;

		internal TFOperation (TFGraph graph, IntPtr handle)
		{
			this.handle = handle;
			this.graph = graph;
		}

		// extern const char * TF_OperationName (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_OperationName (TF_Operation oper);

		/// <summary>
		/// The name for this operation/
		/// </summary>
		/// <value>The name.</value>
		public string Name => handle == IntPtr.Zero ? "<ObjectDisposed>" : TF_OperationName (handle).GetStr ();

		// extern const char * TF_OperationOpType (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_OperationOpType (TF_Operation oper);

		public string OpType => handle == IntPtr.Zero ? "<ObjectDisposedException>" : TF_OperationOpType (handle).GetStr ();

		// extern const char * TF_OperationDevice (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_OperationDevice (TF_Operation oper);

		// public string Device => TF_OperationDevice (handle).GetStr ();

		// extern int TF_OperationNumOutputs (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationNumOutputs (TF_Operation oper);

		/// <summary>
		/// Gets the number of outputs on this operation.
		/// </summary>
		/// <value>The number outputs.</value>
		public int NumOutputs => handle == IntPtr.Zero ? -1 : TF_OperationNumOutputs (handle);


		// extern int TF_OperationOutputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationOutputListLength (TF_Operation oper, string arg_name, TF_Status status);

		public int OutputListLength (string argName, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				TFDisposable.ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var res = TF_OperationOutputListLength (handle, argName, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return res;
		}

		// extern int TF_OperationNumInputs (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationNumInputs (TF_Operation oper);

		/// <summary>
		/// Gets the number of inputs for this operation.
		/// </summary>
		/// <value>The number inputs.</value>
		public int NumInputs => TF_OperationNumInputs (handle);


		// extern int TF_OperationInputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationInputListLength (TF_Operation oper, string arg_name, TF_Status status);

		public int InputListLength (string argName, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				TFDisposable.ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var res = TF_OperationInputListLength (handle, argName, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return res;
		}

		// extern int TF_OperationNumControlInputs (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationNumControlInputs (TF_Operation oper);
		public int NumControlInputs => TF_OperationNumControlInputs (handle);

		// extern int TF_OperationGetControlInputs (TF_Operation *oper, TF_Operation **control_inputs, int max_control_inputs);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationGetControlInputs (TF_Operation oper, TF_Operation control_inputs, int max_control_inputs);

		// extern int TF_OperationNumControlOutputs (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationNumControlOutputs (TF_Operation oper);
		public int NumControlOutputs => TF_OperationNumControlOutputs (handle);

		// extern int TF_OperationGetControlOutputs (TF_Operation *oper, TF_Operation **control_outputs, int max_control_outputs);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationGetControlOutputs (TF_Operation oper, [Out] [MarshalAs (UnmanagedType.LPArray, SizeParamIndex = 2)] IntPtr [] control_outputs, int max_control_outputs);

		TFOperation [] ControlOutputs {
			get {
				var n = NumControlOutputs;
				var arr = new IntPtr [n];
				TF_OperationGetControlOutputs (handle, arr, n);
				var ret = new TFOperation [n];
				for (int i = 0; i < n; i++)
					ret [i] = new TFOperation (graph, arr [i]);
				return ret;
			}
		}

		// extern TF_AttrMetadata TF_OperationGetAttrMetadata (TF_Operation *oper, const char *attr_name, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TFAttributeMetadata TF_OperationGetAttrMetadata (TF_Operation oper, string attr_name, TF_Status status);

		public TFAttributeMetadata GetAttributeMetadata (string attrName, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				TFDisposable.ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var x = TF_OperationGetAttrMetadata (handle, attrName, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return x;
		}

		// extern void TF_OperationGetAttrString (TF_Operation *oper, const char *attr_name, void *value, size_t max_length, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrString (TF_Operation oper, string attr_name, void* value, size_t max_length, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrStringList (TF_Operation *oper, const char *attr_name, void **values, size_t *lengths, int max_values, void *storage, size_t storage_size, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrStringList (TF_Operation oper, string attr_name, void** values, size_t* lengths, int max_values, void* storage, size_t storage_size, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrInt (TF_Operation *oper, const char *attr_name, int64_t *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrInt (TF_Operation oper, string attr_name, long* value, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrIntList (TF_Operation *oper, const char *attr_name, int64_t *values, int max_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrIntList (TF_Operation oper, string attr_name, long* values, int max_values, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrFloat (TF_Operation *oper, const char *attr_name, float *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrFloat (TF_Operation oper, string attr_name, float* value, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrFloatList (TF_Operation *oper, const char *attr_name, float *values, int max_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrFloatList (TF_Operation oper, string attr_name, float* values, int max_values, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrBool (TF_Operation *oper, const char *attr_name, unsigned char *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrBool (TF_Operation oper, string attr_name, byte* value, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrBoolList (TF_Operation *oper, const char *attr_name, unsigned char *values, int max_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrBoolList (TF_Operation oper, string attr_name, byte* values, int max_values, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrType (TF_Operation *oper, const char *attr_name, TF_DataType *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrType (TF_Operation oper, string attr_name, TFDataType* value, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrTypeList (TF_Operation *oper, const char *attr_name, TF_DataType *values, int max_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrTypeList (TF_Operation oper, string attr_name, TFDataType* values, int max_values, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrShape (TF_Operation *oper, const char *attr_name, int64_t *value, int num_dims, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrShape (TF_Operation oper, string attr_name, long* value, int num_dims, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrShapeList (TF_Operation *oper, const char *attr_name, int64_t **dims, int *num_dims, int num_shapes, int64_t *storage, int storage_size, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrShapeList (TF_Operation oper, string attr_name, long** dims, int* num_dims, int num_shapes, long* storage, int storage_size, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrTensorShapeProto (TF_Operation *oper, const char *attr_name, TF_Buffer *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrTensorShapeProto (TF_Operation oper, string attr_name, LLBuffer* value, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrTensorShapeProtoList (TF_Operation *oper, const char *attr_name, TF_Buffer **values, int max_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrTensorShapeProtoList (TF_Operation oper, string attr_name, LLBuffer** values, int max_values, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrTensor (TF_Operation *oper, const char *attr_name, TF_Tensor **value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrTensor (TF_Operation oper, string attr_name, TF_Tensor* value, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrTensorList (TF_Operation *oper, const char *attr_name, TF_Tensor **values, int max_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrTensorList (TF_Operation oper, string attr_name, TF_Tensor* values, int max_values, TF_Status status);
		// TODO:

		// extern void TF_OperationGetAttrValueProto (TF_Operation *oper, const char *attr_name, TF_Buffer *output_attr_value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationGetAttrValueProto (TF_Operation oper, string attr_name, LLBuffer* output_attr_value, TF_Status status);
		// TODO:


		// extern void TF_OperationToNodeDef (TF_Operation *oper, TF_Buffer *output_node_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationToNodeDef (TF_Operation oper, LLBuffer* output_node_def, TF_Status status);

		/// <summary>
		/// Encodes the TFOperation as a protocol buffer payload
		/// </summary>
		/// <returns>The buffer with the encoded operation in the protocol buffer format.</returns>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		/// <remarks>
		/// </remarks>
		public TFBuffer ToNodeDef (TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				TFDisposable.ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var r = new TFBuffer ();
			unsafe
			{
				TF_OperationToNodeDef (handle, r.LLBuffer, cstatus.handle);
			}
			// No need to raise, we can return null in that case.
			if (!cstatus.Ok) {
				r.Dispose ();
				return null;
			}
			return r;
		}

		/// <summary>
		/// Returns the handle to the idx-th output of the operation.
		/// </summary>
		/// <param name="idx">Index of the output in the operation.</param>
		public TFOutput this [int idx] {
			get {
				return new TFOutput (this, idx);
			}
		}
	}

	/// <summary>
	/// Contains options that are used to control how graph importing works.
	/// </summary>
	public class TFImportGraphDefOptions : TFDisposable
	{
		// extern TF_ImportGraphDefOptions * TF_NewImportGraphDefOptions ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_ImportGraphDefOptions TF_NewImportGraphDefOptions ();

		public TFImportGraphDefOptions () : base (TF_NewImportGraphDefOptions ())
		{
		}

		// extern void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions *opts);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteImportGraphDefOptions (TF_ImportGraphDefOptions opts);

		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteImportGraphDefOptions (handle);
		}

		// extern void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions *opts, const char *prefix);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_ImportGraphDefOptionsSetPrefix (TF_ImportGraphDefOptions opts, string prefix);

		public void SetPrefix (string prefix)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();			
			TF_ImportGraphDefOptionsSetPrefix (handle, prefix);
		}

		// extern void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions *opts, const char* src_name, int src_index, TF_Output dst);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_ImportGraphDefOptionsAddInputMapping (TF_ImportGraphDefOptions opts, string src_name, int src_index, TFOutput dst);


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
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			TF_ImportGraphDefOptionsAddInputMapping (handle, srcName, srcIndex, dst);
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		extern static void TF_ImportGraphDefOptionsAddControlDependency (TF_ImportGraphDefOptions opts, TF_Operation oper);

		/// <summary>
		/// Cause the imported graph to have a control dependency on the provided operation.
		/// </summary>
		/// <param name="operation">This operation should exist in the graph being imported to.</param>
		public void AddControlDependency (TFOperation operation)
		{
			if (operation == null)
				throw new ArgumentNullException (nameof (operation));
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			
			TF_ImportGraphDefOptionsAddControlDependency (handle, operation.handle);
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		extern static void TF_ImportGraphDefOptionsAddReturnOutput (TF_ImportGraphDefOptions opts, string oper_name, int index);

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
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			TF_ImportGraphDefOptionsAddReturnOutput (handle, operName, index);
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		extern static int TF_ImportGraphDefOptionsNumReturnOutputs (TF_ImportGraphDefOptions opts);

		/// <summary>
		/// Gets the number return outputs added via AddReturnOutput.
		/// </summary>
		/// <value>The number return outputs.</value>
		public int NumReturnOutputs {
			get {
				if (handle == IntPtr.Zero)
					ObjectDisposedException ();
				return TF_ImportGraphDefOptionsNumReturnOutputs (handle);
			}
		}

		[DllImport (NativeBinding.TensorFlowLibrary)]
		extern static void TF_ImportGraphDefOptionsRemapControlDependency (TF_ImportGraphDefOptions opts, string srcName, TF_Operation dst);

		/// <summary>
		/// Sets any imported nodes with a given control input to have it replaced with an operation
		/// </summary>
		/// <param name="srcName">Node in the graph to be imported.</param>
		/// <param name="destination">References an operation that already exists in the graph being imported.</param>
		/// <remarks>
		/// Set any imported nodes with control input <paramref name="srcName"/> to have that input
		/// replaced with <paramref name="destination"/>. 
		/// </remarks>
		public void RemapControlDependency (string srcName, TFOperation destination)
		{
			if (srcName == null)
				throw new ArgumentNullException (nameof (srcName));
			if (destination == null)
				throw new ArgumentNullException (nameof (destination));
			if (destination.Handle == IntPtr.Zero)
				throw new ObjectDisposedException (nameof (destination));
			TF_ImportGraphDefOptionsRemapControlDependency (handle, srcName, destination.Handle);
		}
	}

	/// <summary>
	/// Drives the execution of a graph
	/// </summary>
	/// <remarks>
	/// <para>
	/// This creates a new context to execute a TFGraph.   You can use the 
	/// constructor to create an empty session, or you can load an existing
	/// model using the <see cref="FromSavedModel"/> static method in this class.
	/// </para>
	/// <para>
	/// To execute operations with the graph, call the <see cref="GetRunner"/>  method
	/// which returns an object that you can use to build the operation by providing
	/// the inputs, requesting the operations that you want to execute and the desired outputs.
	/// </para>
	/// <para>
	/// The <see cref="GetRunner"/> method is a high-level helper function that wraps a
	/// call to the <see cref="Run"/> method which just takes too many parameters that must
	/// be kept in sync.
	/// </para>
	/// </remarks>
	public class TFSession : TFDisposable
	{
		// extern TF_Session * TF_NewSession (TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Session TF_NewSession (TF_Graph graph, TF_SessionOptions opts, TF_Status status);

		/// <summary>
		/// Gets the graph associated with this TensorFlow session.
		/// </summary>
		/// <value>The graph.</value>
		public TFGraph Graph { get; private set; }

		TFSession (IntPtr handle, TFGraph graph) : base (handle) 
		{
			Graph = graph;
		}

		public TFSession (TFGraph graph, TFSessionOptions sessionOptions, TFStatus status = null) : base (IntPtr.Zero)
		{
			Graph = graph;
			var cstatus = TFStatus.Setup (status);
			var h = TF_NewSession (graph.handle, sessionOptions.handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			handle = h;
		}

		public TFSession (TFGraph graph, TFStatus status = null) : base (IntPtr.Zero)
		{
			Graph = graph;
			var cstatus = TFStatus.Setup (status);
			var empty = TFSessionOptions.TF_NewSessionOptions ();
			var h = TF_NewSession (graph.handle, empty, cstatus.handle);
			TFSessionOptions.TF_DeleteSessionOptions (empty);
			cstatus.CheckMaybeRaise (status);
			handle = h;
		}

		public TFSession (TFStatus status = null) : this (new TFGraph (), status)
		{
		}

		// extern TF_Session * TF_LoadSessionFromSavedModel (const TF_SessionOptions *session_options, const TF_Buffer *run_options, const char *export_dir, const char *const *tags, int tags_len, TF_Graph *graph, TF_Buffer *meta_graph_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Session TF_LoadSessionFromSavedModel (TF_SessionOptions session_options, LLBuffer* run_options, string export_dir, string [] tags, int tags_len, TF_Graph graph, LLBuffer* meta_graph_def, TF_Status status);

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
				var h = TF_LoadSessionFromSavedModel (sessionOptions.handle, runOptions.LLBuffer, exportDir, tags, tags.Length, graph.handle, metaGraphDef.LLBuffer, cstatus.handle);

				if (cstatus.CheckMaybeRaise (status)) {
					return new TFSession (h, graph);
				}
			}
			return null;
		}

		// extern void TF_CloseSession (TF_Session *, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_CloseSession (TF_Session session, TF_Status status);

		public void CloseSession (TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			TF_CloseSession (handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
		}

		// extern void TF_DeleteSession (TF_Session *, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteSession (TF_Session session, TF_Status status);

		public void DeleteSession (TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();			
			var cstatus = TFStatus.Setup (status);
			TF_DeleteSession (handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
		}

		internal override void NativeDispose (IntPtr handle)
		{
			using (var s = new TFStatus ()) {
				TF_DeleteSession (handle, s.handle);
			}
		}

		// extern void TF_SessionRun (TF_Session *session, const TF_Buffer *run_options, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Buffer *run_metadata, TF_Status *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SessionRun (TF_Session session, LLBuffer* run_options, TFOutput [] inputs, TF_Tensor [] input_values, int ninputs, TFOutput [] outputs, TF_Tensor [] output_values, int noutputs, TF_Operation [] target_opers, int ntargets, LLBuffer* run_metadata, TF_Status status);


		/// <summary>
		/// Use the runner class to easily configure inputs, outputs and targets to be passed to the session runner.
		/// </summary>
		/// <remarks>
		/// <para>
		/// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
		/// to construct the parameters that will be passed to the TFSession.Run method.
		/// </para>
		/// <para>
		/// Instances of this class are created by calling the GetRunner method on the TFSession.
		/// </para>
		/// <para>
		/// The various methods in this class return an instance to the Runner itsel, to allow
		/// to easily construct chains of execution like this:
		/// </para>
		/// <code>
		/// var result = session.GetRunner ().AddINput (myInput).Fetch (MyOutput).Run ();
		/// </code>
		/// <para>
		/// You do not need to chain the operations, this works just the same:
		/// </para>
		/// <code>
		/// runner = session.GetRunner ();
		/// runner.AddInput(myInput);
		/// runner.Fetch(myOutput);
		/// var results = runner.Run();
		/// </code>
		/// </remarks>
		public class Runner
		{
			List<TFOutput> inputs = new List<TFOutput> (), outputs = new List<TFOutput> ();
			List<TFTensor> inputValues = new List<TFTensor> ();
			List<TFOperation> targets = new List<TFOperation> ();
			TFSession session;

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
			/// Adds an input to the session specified by name, with an optional index in the operation (separated by a colon).
			/// </summary>
			/// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
			/// <param name="input">Incoming port, with an optional index separated by a colon.</param>
			/// <param name="value">Value to assing to the incoming port.</param>
			public Runner AddInput (string input, TFTensor value)
			{
				if (value == null)
					throw new ArgumentNullException (nameof (value));
				inputs.Add (ParseOutput (input));
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


			// Parses user strings that contain both the operation name and an index.
			TFOutput ParseOutput (string operation)
			{
				var p = operation.IndexOf (':');
				if (p != -1 && p != operation.Length - 1){
					var op = operation.Substring (0, p);
					if (int.TryParse (operation.Substring (p + 1), out var idx)){
						return session.Graph [op] [idx];
					}
				}
				return session.Graph [operation] [0];
			}

			/// <summary>
			/// Adds the specified operation names as the ones to be retrieved.
			/// </summary>
			/// <returns>An instance to the runner, so you can easily chain the operations together.</returns>
			/// <param name="targetNames">One or more target names.</param>
			public Runner AddTarget (params string [] targetNames)
			{
				foreach (var tn in targetNames)
					this.targets.Add (session.Graph [tn]);
				return this;
			}

			/// <summary>
			/// Makes the Run method return the index-th output of the tensor referenced by operation.
			/// </summary>
			/// <returns>The instance of runner, to allow chaining operations.</returns>
			/// <param name="operation">The name of the operation in the graph.</param>
			/// <param name="index">The index of the output in the operation.</param>
			public Runner Fetch (string operation, int index)
			{
				var op = session.Graph [operation];
				outputs.Add (op [index]);
				return this;
			}

			/// <summary>
			/// Makes the Run method return the output of the tensor referenced by operation, the operation string can contain the output index.
			/// </summary>
			/// <returns>The instance of runner, to allow chaining operations.</returns>
			/// <param name="operation">The name of the operation in the graph, which might be a simple name, or it might be name:index, 
			/// where the index is the .</param>
			public Runner Fetch (string operation)
			{
				var op = ParseOutput (operation);
				outputs.Add (op);
				return this;
			}

			/// <summary>
			/// Makes the Run method return the output of the tensor referenced by output
			/// </summary>
			/// <returns>The instance of runner, to allow chaining operations.</returns>
			/// <param name="output">The output referencing a specified tensor.</param>
			public Runner Fetch (TFOutput output)
			{
				outputs.Add (output);
				return this;
			}

			/// <summary>
			/// Makes the Run method return the output of all the tensor referenced by outputs.
			/// </summary>
			/// <returns>The instance of runner, to allow chaining operations.</returns>
			/// <param name="outputs">The outputs referencing a specified tensor.</param>
			public Runner Fetch (params TFOutput [] outputs)
			{
				foreach (var output in outputs)
					this.outputs.Add (output);
				return this;
			}

			/// <summary>
			/// Makes the Run method return the output of all the tensor referenced by outputs.
			/// </summary>
			/// <returns>The instance of runner, to allow chaining operations.</returns>
			/// <param name="outputs">The output sreferencing a specified tensor.</param>
			public Runner Fetch (params string [] outputs)
			{
				foreach (var output in outputs)
					this.outputs.Add (ParseOutput (output));
				return this;
			}

			/// <summary>
			/// Protocol buffer encoded block containing the metadata passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
			/// </summary>
			public TFBuffer RunMetadata;

			/// <summary>
			/// Protocol buffer encoded block containing the run options passed to the <see cref="M:TensorFlow.TFSession.Run"/> method.
			/// </summary>
			public TFBuffer RunOptions;

			/// <summary>
			///  Execute the graph fragments necessary to compute all requested fetches.
			/// </summary>
			/// <returns>One TFTensor for each call to Fetch that you made, in the order that you made them.</returns>
			/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
			public TFTensor [] Run (TFStatus status = null)
			{
				return session.Run (inputs.ToArray (), inputValues.ToArray (), outputs.ToArray (), targets.ToArray (), RunMetadata, RunOptions, status);
			}

			/// <summary>
			/// Run the specified operation, by adding it implicity to the output, single return value
			/// </summary>
			/// <param name="operation">The output of the operation.</param>
			/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
			/// <remarks>
			/// This method is a convenience method, and when you call it, it will clear any 
			/// calls that you might have done to Fetch() and use the specified operation to Fetch
			/// instead.
			/// </remarks>
			public TFTensor  Run (TFOutput operation, TFStatus status = null)
			{
				outputs.Clear ();
				Fetch (operation);
				return Run (status) [0];
			}

		}

		/// <summary>
		/// Gets a new runner, this provides a simpler API to prepare the inputs to run on a session
		/// </summary>
		/// <returns>The runner.</returns>
		/// <remarks>
		/// The runner has a simple API that allows developers to call the AddTarget, AddInput, AddOutput and Fetch
		/// to construct the parameters that will be passed to the TFSession.Run method.
		/// 
		/// The Run method will return an array of TFTensor values, one for each invocation to the Fetch method.
		/// </remarks>
		public Runner GetRunner ()
		{
			return new Runner (this);
		}

		/// <summary>
		/// Executes a pipeline given the specified inputs, inputValues, outputs, targetOpers, runMetadata and runOptions.   
		/// A simpler API is available by calling the <see cref="M:GetRunner"/> method which performs all the bookkeeping
		/// necessary.
		/// </summary>
		/// <returns>An array of tensors fetched from the requested outputs.</returns>
		/// <param name="inputs">Inputs nodes.</param>
		/// <param name="inputValues">Input values.</param>
		/// <param name="outputs">Output nodes.</param>
		/// <param name="targetOpers">Target operations to execute.</param>
		/// <param name="runMetadata">Run metadata.</param>
		/// <param name="runOptions">Run options.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public TFTensor [] Run (TFOutput [] inputs, TFTensor [] inputValues, TFOutput [] outputs, TFOperation [] targetOpers = null, TFBuffer runMetadata = null, TFBuffer runOptions = null, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();			
			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (inputValues == null)
				throw new ArgumentNullException (nameof (inputValues));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			int iLen = inputs.Length;
			if (iLen != inputValues.Length)
				throw new ArgumentException ("inputs and inputValues have different lengths", "inputs");
			int oLen = outputs.Length;

			// runOptions and runMetadata might be null
			var cstatus = TFStatus.Setup (status);

			// Create arrays for the unmanaged versions
			var ivals = new IntPtr [iLen];
			for (int i = 0; i < iLen; i++)
				ivals [i] = inputValues [i].handle;

			// I believe this might not be necessary, the output values in TF_SessionRun looks like a write-only result
			var ovals = new IntPtr [outputs.Length];
			IntPtr [] topers = null;
			int tLen = 0;
			if (targetOpers != null) {
				tLen = targetOpers.Length;
				topers = new IntPtr [tLen];
				for (int i = 0; i < tLen; i++)
					topers [i] = targetOpers [i].handle;
			}

			unsafe
			{
				TF_SessionRun (handle, runOptions == null ? null : runOptions.LLBuffer, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, runMetadata == null ? null : runMetadata.LLBuffer, cstatus.handle);
			}
			cstatus.CheckMaybeRaise (status);
			var result = new TFTensor [oLen];
			for (int i = 0; i < oLen; i++) {
				result [i] = new TFTensor (ovals [i]);
			}
			return result;
		}

		// extern void TF_SessionPRunSetup (TF_Session, const TF_Output *inputs, int ninputs, const TF_Output *outputs, int noutputs, const TF_Operation *const *target_opers, int ntargets, const char **handle, TF_Status *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SessionPRunSetup (TF_Session session, TFOutput [] inputs, int ninputs, TFOutput [] outputs, int noutputs, TF_Operation [] target_opers, int ntargets, out IntPtr returnHandle, TF_Status status);

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeletePRunHandle (IntPtr partialRunHandle);

		/// <summary>
		/// Token returned from using one of the Partial Run Setup methods from <see cref="T:TensorFlow.TFSession"/>,
		/// and use this token subsequently for other invocations.
		/// </summary>
		/// <remarks>
		/// Calling Dispose on this object will release the resources associated with setting up 
		/// a partial run.
		/// </remarks>
		public class PartialRunToken : IDisposable
		{
			internal IntPtr token;

			void IDisposable.Dispose ()
			{
				if (token == IntPtr.Zero) {
					TF_DeletePRunHandle (token);
					token = IntPtr.Zero;
				}
			}
		}

		/// <summary>
		/// Prepares the session for a partial run.
		/// </summary>
		/// <returns>A token that can be used to call <see cref="PartialRun"/> repeatedly.   To complete your partial run, you should call Dispose on the resulting method.</returns>
		/// <param name="inputs">Inputs.</param>
		/// <param name="outputs">Outputs.</param>
		/// <param name="targetOpers">Target operations to run.</param>
		/// <param name="status">Status buffer, if specified a status code will be left here, if not specified, a <see cref="T:TensorFlow.TFException"/> exception is raised if there is an error.</param>
		public PartialRunToken PartialRunSetup (TFOutput [] inputs, TFOutput [] outputs, TFOperation [] targetOpers, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();			
			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			if (targetOpers == null)
				throw new ArgumentNullException (nameof (targetOpers));
			
			IntPtr returnHandle;
			var cstatus = TFStatus.Setup (status);
			int tLen = targetOpers.Length;
			var topers = new IntPtr [tLen];
			for (int i = 0; i < tLen; i++)
				topers [i] = targetOpers [i].handle;

			TF_SessionPRunSetup (handle, inputs, inputs.Length, outputs, outputs.Length, topers, tLen, out returnHandle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return new PartialRunToken () { token = returnHandle };
		}

		// extern void TF_SessionPRun (TF_Session *, const char *handle, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Status *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SessionPRun (TF_Session session, IntPtr partialHandle, TFOutput [] inputs, TF_Tensor [] input_values, int ninputs, TFOutput [] outputs, TF_Tensor [] output_values, int noutputs, TF_Operation [] target_opers, int ntargets, TF_Status status);
		public TFTensor [] PartialRun (PartialRunToken token, TFOutput [] inputs, TFTensor [] inputValues, TFOutput [] outputs, TFOperation [] targetOpers, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();			
			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (inputValues == null)
				throw new ArgumentNullException (nameof (inputValues));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			if (targetOpers == null)
				throw new ArgumentNullException (nameof (targetOpers));
			int iLen = inputs.Length;
			if (iLen != inputValues.Length)
				throw new ArgumentException ("inputs and inputValues have different lengths", "inputs");
			int oLen = outputs.Length;

			// runOptions and runMetadata might be null
			var cstatus = TFStatus.Setup (status);

			// Create arrays for the unmanaged versions
			var ivals = new IntPtr [iLen];
			for (int i = 0; i < iLen; i++)
				ivals [i] = inputValues [i].handle;
			var ovals = new IntPtr [oLen];
			int tLen = targetOpers.Length;
			var topers = new IntPtr [tLen];
			for (int i = 0; i < tLen; i++)
				topers [i] = targetOpers [i].handle;

			unsafe
			{
				TF_SessionPRun (handle, token.token, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, cstatus.handle);
			}
			cstatus.CheckMaybeRaise (status);

			var result = new TFTensor [oLen];
			for (int i = 0; i < oLen; i++) {
				result [i] = new TFTensor (ovals [i]);
			}
			return result;
		}
	}

	public class TFLibrary : TFDisposable {
		// extern TF_Library * TF_LoadLibrary (const char *library_filename, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Library TF_LoadLibrary (string library_filename, TF_Status  status);

		TFLibrary (IntPtr handle) : base (handle) { }

		public static TFLibrary FromFile (string libraryFile, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var h = TF_LoadLibrary (libraryFile, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			return new TFLibrary (h);
		}

		// extern TF_Buffer TF_GetOpList (TF_Library *lib_handle);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer TF_GetOpList (TF_Library lib_handle);


		/// <summary>
		/// Retrieves the ProtocolBuffer describing the available operations in
		/// the loaded TensorFlow library.
		/// </summary>
		/// <returns>The buffer contains a ProtocolBuffer encoded payload, you need a ProtocolBuffer reader to process the contents.</returns>
		TFBuffer GetOpList ()
		{
			return new TFBuffer (TF_GetOpList (handle).data);
		}

		// extern void TF_DeleteLibraryHandle (TF_Library *lib_handle);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteLibraryHandle (TF_Library lib_handle);

		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteLibraryHandle (handle);
		}
	}

	/// <summary>
	/// The data type for a specific tensor.
	/// </summary>
	/// <remarks>
	/// Tensors have uniform data types, all the elements of the tensor are of this
	/// type and they dictate how TensorFlow will treat the data stored.   
	/// </remarks>
	public enum TFDataType : uint
	{
		/// <summary>
		/// Single precission floatint point, 32-bits (C# float)
		/// </summary>
		Float = 1,
		/// <summary>
		/// Double precission floatint point, 64-bits (C# double)
		/// </summary>
		Double = 2,
		/// <summary>
		/// 32-bit signed integers (C# int)
		/// </summary>
		Int32 = 3,
		/// <summary>
		/// 8 bit unsigned integers (C# byte)
		/// </summary>
		UInt8 = 4,
		/// <summary>
		/// 16-bit signed integers (C# short)
		/// </summary>
		Int16 = 5,
		/// <summary>
		/// 8-bit signed integers (C# sbyte)
		/// </summary>
		Int8 = 6,
		/// <summary>
		/// Binary blob
		/// </summary>
		String = 7,
		/// <summary>
		/// Single precission complex numbers (32-bit floats)
		/// </summary>
		Complex64 = 8,
		/// <summary>
		/// 32-bit float based complex numbers
		/// </summary>
		Complex = 8,
		/// <summary>
		/// 64-bit signed integers (C# long)
		/// </summary>
		Int64 = 9,
		Bool = 10,
		/// <summary>
		/// Quantized 8-bit signed integer
		/// </summary>
		QInt8 = 11,
		/// <summary>
		/// Quantized 8-bit unsigned integer
		/// </summary>
		QUInt8 = 12,
		/// <summary>
		/// Quantized 32-bit signed integer
		/// </summary>
		QInt32 = 13,
		/// <summary>
		/// Float32 truncated to 16 bits.  Only for cast operations.
		/// </summary>
		BFloat16 = 14,
		/// <summary>
		/// Quantized 16-bit signed integer
		/// </summary>
		QInt16 = 15,
		/// <summary>
		/// Quantized 16-bit unsigned integer
		/// </summary>
		QUInt16 = 16,
		/// <summary>
		/// 16-bit unsigned integers (C# long)
		/// </summary>
		UInt16 = 17,
		/// <summary>
		/// Double precission complex numbers (32-bit floats)
		/// </summary>
		Complex128 = 18,
		Half = 19,
		Resource = 20
	}

	/// <summary>
	/// Status code for invoking a tensorflow operation.
	/// </summary>
	public enum TFCode : uint
	{
		Ok = 0,
		Cancelled = 1,
		Unknown = 2,
		InvalidArgument = 3,
		DeadlineExceeded = 4,
		NotFound = 5,
		AlreadyExists = 6,
		PermissionDenied = 7,
		Unauthenticated = 16,
		ResourceExhausted = 8,
		FailedPrecondition = 9,
		Aborted = 10,
		OutOfRange = 11,
		Unimplemented = 12,
		Internal = 13,
		Unavailable = 14,
		DataLoss = 15
	}

	[StructLayout (LayoutKind.Sequential)]
	public struct TFInput
	{
		public unsafe TF_Operation Operation;
		public int Index;

		// extern TF_Output TF_OperationInput (TF_Input oper_in);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern TFOutput TF_OperationInput (TFInput oper_in);

		public TFOutput GetOutput (TFInput operIn)
		{
			return TF_OperationInput (operIn);
		}

		// extern TF_DataType TF_OperationInputType (TF_Input oper_in);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern TFDataType TF_OperationInputType (TFInput oper_in);

		public TFDataType InputType => TF_OperationInputType (this);

	}

	/// <summary>
	/// Represents a specific output of an operation on a tensor.
	/// </summary>
	/// <remarks>
	/// <para>
	/// TFOutput objects represent one of the outputs of an operation in the graph
	/// (TFGraph).  Outputs have a data type, and eventually a shape that you can 
	/// retrieve by calling the <see cref="M:TensorFlow.TFGraph.GetShape"/> method.
	/// </para>
	/// <para>
	/// These can be passed as an input argument to a function for adding operations 
	/// to a graph, or to the TFSession's Run and GetRunner method as values to be
	/// fetched.
	/// </para>
	/// </remarks>
	[StructLayout (LayoutKind.Sequential)]
	public struct TFOutput
	{
		unsafe TF_Operation LLOperation;
		public int Index;

		// extern int TF_OperationOutputNumConsumers (TF_Output oper_out);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern int TF_OperationOutputNumConsumers (TFOutput oper_out);

		/// <summary>
		/// Gets the number consumers.
		/// </summary>
		/// <value>The number consumers.</value>
		/// <remarks>
		/// This number can change when new operations are added to the graph.
		/// </remarks>
		public int NumConsumers => TF_OperationOutputNumConsumers (this);

		// extern TF_DataType TF_OperationOutputType (TF_Output oper_out);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern TFDataType TF_OperationOutputType (TFOutput oper_out);

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
			LLOperation = operation.handle;
			Index = index;
		}

		/// <summary>
		/// Initializes a new TFOutput instance from another TFOutput
		/// </summary>
		/// <param name="operation">The other TFOutput that is having its operation attached.</param>
		/// <param name="index">The index of the output within the operation, if not specified, it defaults to zero.</param>
		public TFOutput (TFOutput output, int index = 0)
		{
			if (output.LLOperation == null)
				throw new ArgumentNullException ("Outputs does not have a valid operation pointer");
			LLOperation = output.LLOperation;
			Index = index;
		}

		// extern int TF_OperationOutputConsumers (TF_Output oper_out, TF_Input *consumers, int max_consumers);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationOutputConsumers (TFOutput oper_out, TFInput* consumers, int max_consumers);

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

	/// <summary>
	/// Low-level: Enumeration describing the types of a metadata attribute
	/// </summary>
	public enum TFAttributeType : uint
	{
		/// <summary>
		/// The type of the attribute is a string
		/// </summary>
		String = 0,

		/// <summary>
		/// The type of the attribute is an int.
		/// </summary>
		Int = 1,

		/// <summary>
		/// The type of the attribute is a float
		/// </summary>
		Float = 2,

		/// <summary>
		/// The type of the attribute is a bool.
		/// </summary>
		Bool = 3,

		/// <summary>
		/// The type of the attribute is a type.
		/// </summary>
		Type = 4,

		/// <summary>
		/// The type of the attribute is a tensor shape
		/// </summary>
		Shape = 5,

		/// <summary>
		/// The type of the attribute is a tensor
		/// </summary>
		Tensor = 6,

		/// <summary>
		/// The type of the attribute is a placeholder
		/// </summary>
		Placeholder = 7,

		/// <summary>
		/// The type of the attribute is a function
		/// </summary>
		Func = 8
	}

	/// <summary>
	/// Low-level: this describes the tensorflow type information for an attribute in the low-level attributes used by operations.
	/// </summary>
	/// <remarks>
	/// This is a low-level operation returned by the <see cref="M:TensorFlow.TFOperation.GetAttributeMetadata"/>.
	/// This is included for completeness, but is not generally used from C#, as you have access to the high-level
	/// bindings in the <see cref="T:TensorFlow.TFGraph"/> type.
	/// </remarks>
	[StructLayout (LayoutKind.Sequential)]
	public struct TFAttributeMetadata
	{
		byte isList;
		public bool IsList => isList != 0;
		public long ListSize;
		public TFAttributeType Type;
		public long TotalSize;

		public override string ToString ()
		{
			return string.Format ($"[TFAttributeMetadata IsList={IsList} ListSize={ListSize} Type={Type} TotalSize={TotalSize}]");
		}
	}

	/// <summary>
	/// Represents the shape of a tensor
	/// </summary>
	/// <remarks>
	/// <para>
	/// The shapes can be created by calling the constructor with the number of dimensions
	/// in the shape.   The null value is used to specify that the shape is unknown,
	/// an empty array is used to create a scalar, and other values are used to specify
	/// the number of dimensions.
	/// </para>
	/// <para>
	/// For the Unknown case, you can use <see cref="P:TensorFlor.TFShape.Unknown"/>, for
	/// scalars, you can use the <see cref="P:TensorFlor.TFShape.Scalar"/> shape.
	/// </para>
	/// <para>
	/// To create a 2-element vector, use:
	/// new TFShape (2)
	/// </para>
	/// <para>
	/// To create a 2x3 matrix, use:
	/// new TFShape (2, 3)
	/// </para>
	/// <para>
	/// To create a shape with an unknown number of elements, you can pass the value
	/// -1.  This is typically used to indicate the shape of tensors that represent a
	/// variable-sized batch of values.
	/// </para>
	/// <para>
	/// To create a matrix with 4 columns and an unknown number of rows:
	/// var batch = new TFShape (-1, 4)
	/// </para>
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
		public static TFShape Scalar => new TFShape (new long [0]);

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
			this.dims = args;
		}

		/// <summary>
		/// Gets the length of the specified dimension in the tensor
		/// </summary>
		/// <returns>The length, -1 for shapes that have an unknown dimension.</returns>
		/// <param name="dimension">Dimension.</param>
		public int GetLength (int dimension) => dims == null ? -1 : dims.GetLength (dimension);

		/// <summary>
		/// Number of dimensions represented by this shape.
		/// </summary>
		/// <value>The number dimensions, -1 if the number of dimensions is unknown, 0 if the shape represent a scalar, 1 for a vector, 2 for a matrix and so on..</value>
		public int NumDimensions => dims == null ? -1 : dims.Length;

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
			if (dims == null)
				return null;
			
			var ret = (long [])dims.Clone ();
			return ret;
		}

		/// <summary>
		/// Returns the shape as an array
		/// </summary>
		/// <returns>null if the shape represents an unknown shape, otherwise an array with N elements, one per dimension, and each element can be either -1 (if the dimension size is unspecified) or the size of the dimension.</returns>
		public int [] ToIntArray ()
		{
			if (dims == null)
				return null;

			var ret = new int [dims.Length];
			for (int i = 0; i < dims.Length; i++) {
				checked {
					ret [i] = (int) dims [i];
				}
			}
			return ret;
		}

		/// <summary>
		/// Gets a value indicating whether one of the dimensions <see cref="T:TensorFlow.TFShape"/> in the shape is larger than Int32.MaxValue.
		/// </summary>
		/// <value><c>true</c> if is long array; otherwise, <c>false</c>.</value>
		public bool IsLongArray {
			get {
				foreach (var l in dims)
					if (l > Int32.MaxValue)
						return true;

				return false;
			}
		}

		public override string ToString ()
		{
			if (dims == null)
				return "unknown";
			return "[" + String.Join (", ", dims.Select (x => x == -1 ? "?" : x.ToString ())) + "]";
		}

		/// <summary>
		/// Gets the dimensions for the specified index.
		/// </summary>
		/// <param name="idx">Index.</param>
		public long this [int idx] => dims [idx];

		/// <summary>
		/// Returns the shape as a 1-dimensional tensor with each element corresponding to the specified shape dimension.
		/// </summary>
		/// <returns>The tensor.</returns>
		public TFTensor AsTensor ()
		{
			return new TFTensor (ToIntArray ());
		}

		/// <summary>
		/// Adds a <see cref="TensorFlow.TFShape"/> to a <see cref="TensorFlow.TFShape"/>, yielding a shape made up of the concatenation of the first and the second shapes.
		/// </summary>
		/// <param name="left">The first <see cref="TensorFlow.TFShape"/> to add.</param>
		/// <param name="right">The second <see cref="TensorFlow.TFShape"/> to add.</param>
		/// <returns>The <see cref="T:TensorFlow.TFShape"/> that is the sum of the values of <c>left</c> and <c>right</c>.</returns>
		public static TFShape operator + (TFShape left, TFShape right)
		{
			if (left == null)
				return right;
			if (right == null)
				return left;

			var full = new long [left.dims.Length + right.dims.Length];
			Array.Copy (left.dims, full, left.dims.Length);
			Array.Copy (right.dims, 0, full, left.dims.Length, right.dims.Length);
			return new TFShape (full);
		}
	}



}
