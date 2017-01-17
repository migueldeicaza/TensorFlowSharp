//
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
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Globalization;

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

using size_t = System.IntPtr;

namespace TensorFlow
{
	static partial class NativeBinding
	{
		public const string TensorFlowLibrary = "libtensorflow";

		internal static string GetStr (this IntPtr x) => Marshal.PtrToStringAnsi (x);
	}

	public static class TFCore {
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_Version ();

		public static string Version => TF_Version ().GetStr ();

		// extern size_t TF_DataTypeSize (TF_DataType dt);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern IntPtr TF_DataTypeSize (TFDataType dt);

		public static long GetDataTypeSize (TFDataType dt) => (long)TF_DataTypeSize (dt);

	}

	public abstract class TFDisposable : IDisposable
	{
		internal IntPtr handle;

		public TFDisposable (IntPtr handle)
		{
			this.handle = handle;
		}

		public void Dispose ()
		{
			Dispose (true);
			GC.SuppressFinalize (this);
		}

		~TFDisposable ()
		{
			Dispose (false);
		}

		internal abstract void NativeDispose (IntPtr handle);

		public virtual void Dispose (bool disposing)
		{
			if (disposing) {
				if (handle != IntPtr.Zero)
					NativeDispose (handle);
				handle = IntPtr.Zero;
			}
		}
	}

	public class TFException : Exception {
		public TFException (string message) : base (message) { }
	}

	public class TFStatus : TFDisposable
	{
		// extern TF_Status * TF_NewStatus ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Status TF_NewStatus ();

		[ThreadStatic] public static TFStatus Default = new TFStatus ();

		public TFStatus () : base (TF_NewStatus ())
		{
		}

		// extern void TF_DeleteStatus (TF_Status *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteStatus (TF_Status status);

		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteStatus (handle);
		}


		// extern void TF_SetStatus (TF_Status *s, TF_Code code, const char *msg);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetStatus (TF_Status s, TFCode code, string msg);

		public void SetStatusCode (TFCode code, string msg)
		{
			TF_SetStatus (handle, code, msg);
		}

		// extern TF_Code TF_GetCode (const TF_Status *s);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TFCode TF_GetCode (TF_Status s);

		public TFCode StatusCode {
			get {
				return TF_GetCode (handle);
			}
		}

		// extern const char * TF_Message (const TF_Status *s);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_Message (TF_Status s);

		public string StatusMessage => TF_Message (handle).GetStr ();

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
		internal bool MaybeRaise (TFStatus incomingStatus)
		{
			if (incomingStatus == null) {
				if (StatusCode != TFCode.Ok) {
					var e = new TFException (StatusMessage);
					Dispose ();
					throw e;
				}
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

	[StructLayout (LayoutKind.Sequential)]
	internal struct LLBuffer
	{
		internal IntPtr data;
		internal size_t length;
		internal IntPtr data_deallocator;
	}	

	// TODO: the string ctor
	public class TFBuffer : TFDisposable
	{
		// extern TF_Buffer * TF_NewBufferFromString (const void *proto, size_t proto_len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer* TF_NewBufferFromString (IntPtr proto, IntPtr proto_len);

		// extern TF_Buffer * TF_NewBuffer ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer* TF_NewBuffer ();

		unsafe public TFBuffer () : base ((IntPtr)TF_NewBuffer ())
		{
		}

		unsafe public TFBuffer (IntPtr buffer, long size) : base ((IntPtr)TF_NewBuffer ())
		{
			LLBuffer* buf = (LLBuffer*)handle;
			buf->data = buffer;
			buf->length = (size_t)size;
			buf->data_deallocator = IntPtr.Zero;
		}

		internal static void FreeBlock (IntPtr data, IntPtr lenght)
		{
			Marshal.FreeHGlobal (data);
		}

		static IntPtr FreeBufferFunc;

		static TFBuffer ()
		{
			FreeBufferFunc = Marshal.GetFunctionPointerForDelegate<Action<IntPtr,IntPtr>> (FreeBlock);
		}

		// This constructor makes a copy of the data
		public TFBuffer (byte [] buffer, int start, int count) : this ()
		{
			if (start < 0 || start >= buffer.Length)
				throw new ArgumentException ("start");
			if (count < 0 || count > buffer.Length - start)
				throw new ArgumentException ("count");
			unsafe
			{
				LLBuffer* buf = LLBuffer;
				buf->data = Marshal.AllocHGlobal (count);
				Marshal.Copy (buffer, start, buf->data, count);
				buf->length = (size_t)count;
				buf->data_deallocator = FreeBufferFunc;
			}
		}

		unsafe internal LLBuffer * LLBuffer => (LLBuffer*)handle;

		// extern void TF_DeleteBuffer (TF_Buffer *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteBuffer (LLBuffer* buffer);

		internal override void NativeDispose (TF_Status handle)
		{
			unsafe { TF_DeleteBuffer ((LLBuffer*)handle); }
		}

		// extern TF_Buffer TF_GetBuffer (TF_Buffer *buffer);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer TF_GetBuffer (LLBuffer *buffer);
	}

	// TODO: More convenience constructors
	// TODO: 
	public class TFTensor : TFDisposable
	{
		// extern TF_Tensor * TF_NewTensor (TF_DataType, const int64_t *dims, int num_dims, void *data, size_t len, void (* deallocator)(void *, size_t, void *), void *deallocator_arg);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_NewTensor (TFDataType dataType, [In] ref long [] dims, int num_dims, IntPtr data, size_t len, Action<IntPtr, size_t, IntPtr> deallocator, IntPtr deallocator_arg);

		internal static void FreeTensorData (IntPtr data, IntPtr len, IntPtr closure)
		{
			Marshal.FreeHGlobal (data);
		}

		// TODO:
		// Great opportunity to create overloads based on the TFDataType that take strongly typed data arrays instead of bytes

		public TFTensor (TFDataType dataType, long [] dims, byte [] data, int start, int count) : base (IntPtr.Zero)
		{
			if (dims == null)
				throw new ArgumentNullException ("dims");

			IntPtr dataCopy = Marshal.AllocHGlobal (count);
			try {
				Marshal.Copy (data, start, dataCopy, count);
			} catch {
				Marshal.FreeHGlobal (dataCopy);
				throw;
			}
			handle = TF_NewTensor (dataType, ref dims, dims.Length, dataCopy, (IntPtr)count, FreeTensorData, IntPtr.Zero);
		}

		internal override void NativeDispose (TF_Status handle)
		{
			TF_DeleteTensor (handle);
		}

		// extern TF_Tensor * TF_AllocateTensor (TF_DataType, const int64_t *dims, int num_dims, size_t len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_AllocateTensor (TFDataType dataType, [In] ref long [] dims, int num_dims, IntPtr len);

		public TFTensor (TFDataType dataType, long [] dims, int size) : base (IntPtr.Zero)
		{
			if (dims == null)
				throw new ArgumentNullException ("dims");
			handle = TF_AllocateTensor (dataType, ref dims, dims.Length, (IntPtr)size);
		}

		// extern void TF_DeleteTensor (TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteTensor (TF_Tensor tensor);

		// extern TF_DataType TF_TensorType (const TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TFDataType TF_TensorType (TF_Tensor tensor);

		public TFDataType TensorType => TF_TensorType (handle);

		// extern int TF_NumDims (const TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_NumDims (TF_Tensor tensor);

		public int NumDims => TF_NumDims (handle);

		// extern int64_t TF_Dim (const TF_Tensor *tensor, int dim_index);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe long TF_Dim (TF_Tensor tensor, int dim_index);

		public long GetTensorDimension (int dimIndex)
		{
			return TF_Dim (handle, dimIndex);
		}

		// extern size_t TF_TensorByteSize (const TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe size_t TF_TensorByteSize (TF_Tensor tensor);

		public IntPtr ByteSize => TF_TensorByteSize (handle);

		// extern void * TF_TensorData (const TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_TensorData (TF_Tensor tensor);

		public IntPtr Data => TF_TensorData (handle);
	}

	// TODO: All these
	static partial class NativeBinding
	{
		// extern size_t TF_StringEncode (const char *src, size_t src_len, char *dst, size_t dst_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe size_t TF_StringEncode (sbyte* src, size_t src_len, sbyte* dst, size_t dst_len, TF_Status status);

		// extern size_t TF_StringDecode (const char *src, size_t src_len, const char **dst, size_t *dst_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe size_t TF_StringDecode (sbyte* src, size_t src_len, sbyte** dst, size_t* dst_len, TF_Status status);

		// extern size_t TF_StringEncodedSize (size_t len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern size_t TF_StringEncodedSize (size_t len);
	}

	public class TFSessionOptions : TFDisposable
	{
		// extern TF_SessionOptions * TF_NewSessionOptions ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_SessionOptions TF_NewSessionOptions ();

		public TFSessionOptions () : base (TF_NewSessionOptions ()) { }

		// extern void TF_DeleteSessionOptions (TF_SessionOptions *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteSessionOptions (TF_SessionOptions options);
		internal override void NativeDispose (TF_Status handle)
		{
			TF_DeleteSessionOptions (handle);
		}

		// extern void TF_SetTarget (TF_SessionOptions *options, const char *target);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetTarget (TF_SessionOptions options, string target);
		public void SetTarget (string target)
		{
			TF_SetTarget (handle, target);
		}

		// extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetConfig (TF_SessionOptions options, IntPtr proto, size_t proto_len, TF_Status status);


		public void SetConfig (IntPtr protoData, int length, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);

			TF_SetConfig (handle, protoData, (IntPtr)length, cstatus.handle);
			cstatus.MaybeRaise (status);
		}

	}

	public class TFGraph : TFDisposable
	{
		// extern TF_Graph * TF_NewGraph ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Graph TF_NewGraph ();

		public TFGraph () : base (TF_NewGraph ())
		{
		}

		// extern void TF_DeleteGraph (TF_Graph *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteGraph (TF_Graph graph);
		internal override void NativeDispose (TF_Status handle)
		{
			TF_DeleteGraph (handle);
		}

		// extern void TF_GraphSetTensorShape (TF_Graph *graph, TF_Output output, const int64_t *dims, const int num_dims, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphSetTensorShape (TF_Graph graph, TFOutput output, ref long [] dims, int num_dims, TF_Status status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphSetTensorShape (TF_Graph graph, TFOutput output, IntPtr dims, int num_dims, TF_Status status);

		public void SetTensorShape (TFOutput output, long [] dims, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			if (dims == null)
				TF_GraphSetTensorShape (handle, output, IntPtr.Zero, 0, cstatus.handle);
			else
				TF_GraphSetTensorShape (handle, output, ref dims, dims.Length, cstatus.handle);
			cstatus.MaybeRaise (status);
		}

		// extern int TF_GraphGetTensorNumDims (TF_Graph *graph, TF_Output output, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_GraphGetTensorNumDims (TF_Graph graph, TFOutput output, TF_Status status);

		public int GetTensorNumDims (TFOutput output, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var code = TF_GraphGetTensorNumDims (handle, output, cstatus.handle);
			cstatus.MaybeRaise (status);
			return code;
		}

		// extern void TF_GraphGetTensorShape (TF_Graph *graph, TF_Output output, int64_t *dims, int num_dims, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphGetTensorShape (TF_Graph graph, TFOutput output, ref long [] dims, int num_dims, TF_Status status);

		public void GetTensorShape (TFOutput output, long [] dims, TFStatus status = null)
		{
			if (dims == null)
				throw new ArgumentNullException ("dims");
			var cstatus = TFStatus.Setup (status);
			TF_GraphGetTensorShape (handle, output, ref dims, dims.Length, cstatus.handle);
			cstatus.MaybeRaise (status);
		}

		// extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphToGraphDef (TF_Graph graph, LLBuffer* output_graph_def, TF_Status status);

		public void ToGraphDef (TFBuffer outputGraphDef, TFStatus status = null)
		{
			if (outputGraphDef == null)
				throw new ArgumentNullException (nameof (outputGraphDef));
			
			var cstatus = TFStatus.Setup (status);
			unsafe
			{
				TF_GraphToGraphDef (handle, outputGraphDef.LLBuffer, cstatus.handle);
			}
			cstatus.MaybeRaise (status);
		}

		// extern void TF_GraphImportGraphDef (TF_Graph *graph, const TF_Buffer *graph_def, const TF_ImportGraphDefOptions *options, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphImportGraphDef (TF_Graph graph, LLBuffer* graph_def, TF_ImportGraphDefOptions options, TF_Status status);

		public void ImportGraphDef (TFBuffer graphDef)
		{
			// TODO
		}

	
	}

	//
	// TF_Operation and TF_OperationDescription are the same, they just represent two separate states
	// on the cycle of the object. 
	//
	public class TFOperation : TFDisposable
	{
		bool owns;

		// extern TF_OperationDescription * TF_NewOperation (TF_Graph *graph, const char *op_type, const char *oper_name);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_OperationDescription TF_NewOperation (TF_Graph graph, string opType, string oper_name);

		internal TFOperation (IntPtr handle, bool owns) : base (handle)
		{
			this.owns = owns;
		}

		public TFOperation (TFGraph graph, string opType, string operName) : base (IntPtr.Zero)
		{
			if (graph == null)
				throw new ArgumentNullException ("graph");
			handle = TF_NewOperation (graph.handle, opType, operName);
			owns = true;
		}

		internal override void NativeDispose (TF_Status handle)
		{
			if (owns) {
				// TODO: Nothing yet
			}

			// else - nothing
		}

		// extern void TF_SetDevice (TF_OperationDescription *desc, const char *device);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetDevice (TF_OperationDescription desc, string device);

		public void SetDevice (string device)
		{
			if (device == null)
				throw new ArgumentNullException ("device");
			TF_SetDevice (handle, device);
		}

		// extern void TF_AddInput (TF_OperationDescription *desc, TF_Output input);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddInput (TF_OperationDescription desc, TFOutput input);

		public void AddInput (TFOutput input)
		{
			TF_AddInput (handle, input);
		}

		// extern void TF_AddInputList (TF_OperationDescription *desc, const TF_Output *inputs, int num_inputs);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddInputList (TF_OperationDescription desc, TFOutput [] inputs, int num_inputs);

		public void AddInputs (TFOutput [] inputs)
		{
			if (inputs == null)
				throw new ArgumentNullException ("inputs");
			TF_AddInputList (handle, inputs, inputs.Length);
		}

		// extern void TF_AddControlInput (TF_OperationDescription *desc, TF_Operation *input);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddControlInput (TF_OperationDescription desc, TF_Operation input);

		public void AddControlInput (TFOperation input)
		{
			if (input == null)
				throw new ArgumentNullException ("input");

			TF_AddControlInput (handle, input.handle);
		}

		// extern void TF_ColocateWith (TF_OperationDescription *desc, TF_Operation *op);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_ColocateWith (TF_OperationDescription desc, TF_Operation op);

		public void ColocateWith (TFOperation op)
		{
			if (op == null)
				throw new ArgumentNullException ("op");
			TF_ColocateWith (handle, op.handle);
		}

		// extern void TF_SetAttrString (TF_OperationDescription *desc, const char *attr_name, const void *value, size_t length);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrString (TF_OperationDescription desc, string attr_name, IntPtr value, size_t length);

		public void SetAttr (string attrName, string value)
		{
			var bytes = Encoding.UTF8.GetBytes (value);
			var buf = Marshal.AllocHGlobal (bytes.Length + 1);
			Marshal.Copy (bytes, 0, buf, bytes.Length);

			TF_SetAttrString (handle, attrName, buf, (IntPtr)bytes.Length);
		}

		// extern void TF_SetAttrStringList (TF_OperationDescription *desc, const char *attr_name, const void *const *values, const size_t *lengths, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrStringList (TF_OperationDescription desc, string attr_name, void** values, size_t* lengths, int num_values);
		// TODO: SetAttrStringList

		// extern void TF_SetAttrInt (TF_OperationDescription *desc, const char *attr_name, int64_t value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrInt (TF_OperationDescription desc, string attr_name, long value);

		public void SetAttr (string attrName, long value)
		{
			TF_SetAttrInt (handle, attrName, value);
		}

		// extern void TF_SetAttrIntList (TF_OperationDescription *desc, const char *attr_name, const int64_t *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrIntList (TF_OperationDescription desc, string attr_name, long* values, int num_values);
		// TODO: Above

		// extern void TF_SetAttrFloat (TF_OperationDescription *desc, const char *attr_name, float value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrFloat (TF_OperationDescription desc, string attr_name, float value);

		public void SetAttr (string attrName, float value)
		{
			TF_SetAttrFloat (handle, attrName, value);
		}

		// extern void TF_SetAttrFloatList (TF_OperationDescription *desc, const char *attr_name, const float *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrFloatList (TF_OperationDescription desc, string attr_name, float* values, int num_values);
		// TODO: above

		// extern void TF_SetAttrBool (TF_OperationDescription *desc, const char *attr_name, unsigned char value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrBool (TF_OperationDescription desc, string attr_name, byte value);

		public void SetAttr (string attrName, bool value)
		{
			TF_SetAttrBool (handle, attrName, (byte)(value ? 1 : 0));
		}

		// extern void TF_SetAttrBoolList (TF_OperationDescription *desc, const char *attr_name, const unsigned char *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrBoolList (TF_OperationDescription desc, string attr_name, byte* values, int num_values);
		// TODO:

		// extern void TF_SetAttrType (TF_OperationDescription *desc, const char *attr_name, TF_DataType value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrType (TF_OperationDescription desc, string attr_name, TFDataType value);

		public void SetAttrType (string attrName, TFDataType dataType)
		{
			TF_SetAttrType (handle, attrName, dataType);
		}

		// extern void TF_SetAttrTypeList (TF_OperationDescription *desc, const char *attr_name, const TF_DataType *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTypeList (TF_OperationDescription desc, string attr_name, TFDataType* values, int num_values);
		// TODO:

		// extern void TF_SetAttrShape (TF_OperationDescription *desc, const char *attr_name, const int64_t *dims, int num_dims);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, long [] dims, int num_dims);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, IntPtr dims, int num_dims);

		public void SetAttrShape (string attrName, long [] dims)
		{
			if (dims == null)
				TF_SetAttrShape (handle, attrName, null, -1);
			else
				TF_SetAttrShape (handle, attrName, dims, dims.Length);
		}

		// extern void TF_SetAttrShapeList (TF_OperationDescription *desc, const char *attr_name, const int64_t *const *dims, const int *num_dims, int num_shapes);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShapeList (TF_OperationDescription desc, string attr_name, long** dims, int* num_dims, int num_shapes);
		// TODO:

		// extern void TF_SetAttrTensorShapeProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorShapeProto (TF_OperationDescription desc, string attr_name, void* proto, size_t proto_len, TF_Status status);
		// TODO:

		// extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription *desc, const char *attr_name, const void *const *protos, const size_t *proto_lens, int num_shapes, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, void** protos, size_t* proto_lens, int num_shapes, TF_Status status);
		// TODO:

		// extern void TF_SetAttrTensor (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensor (TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status);

		public void SetAttr (string attrName, TFTensor tensor, TFStatus status = null)
		{
			if (tensor == null)
				throw new ArgumentNullException ("tensor");
			var cstatus = TFStatus.Setup (status);

			TF_SetAttrTensor (handle, attrName, tensor.handle, cstatus.handle);
			cstatus.MaybeRaise (status);
		}

		// extern void TF_SetAttrTensorList (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *const *values, int num_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorList (TF_OperationDescription desc, string attr_name, TF_Tensor values, int num_values, TF_Status status);
		// TODO:

		// extern void TF_SetAttrValueProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrValueProto (TF_OperationDescription desc, string attr_name, void* proto, size_t proto_len, TF_Status status);
		// TODO:

		// extern TF_Operation * TF_FinishOperation (TF_OperationDescription *desc, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_FinishOperation (TF_OperationDescription desc, TF_Status status);

		public void FinishOperation (TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			TF_FinishOperation (handle, cstatus.handle);
			cstatus.MaybeRaise (status);
		}

		// extern const char * TF_OperationName (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_OperationName (TF_Operation oper);

		public string Name => TF_OperationName (handle).GetStr ();

		// extern const char * TF_OperationOpType (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_OperationOpType (TF_Operation oper);

		public string OpType => TF_OperationOpType (handle).GetStr ();

		// extern const char * TF_OperationDevice (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_OperationDevice (TF_Operation oper);

		public string Device => TF_OperationDevice (handle).GetStr ();

		// extern int TF_OperationNumOutputs (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationNumOutputs (TF_Operation oper);

		public int NumOutputs => TF_OperationNumOutputs (handle);


		// extern int TF_OperationOutputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationOutputListLength (TF_Operation oper, string arg_name, TF_Status status);

		public int OutputListLength (string argName, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var res = TF_OperationOutputListLength (handle, argName, cstatus.handle);
			cstatus.MaybeRaise (status);
			return res;
		}

		// extern int TF_OperationNumInputs (TF_Operation *oper);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationNumInputs (TF_Operation oper);

		public int NumInputs => TF_OperationNumInputs (handle);


		// extern int TF_OperationInputListLength (TF_Operation *oper, const char *arg_name, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationInputListLength (TF_Operation oper, string arg_name, TF_Status status);

		public int InputListLength (string argName, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var res = TF_OperationInputListLength (handle, argName, cstatus.handle);
			cstatus.MaybeRaise (status);
			return res;
		}

		// extern int TF_OperationOutputConsumers (TF_Output oper_out, TF_Input *consumers, int max_consumers);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe int TF_OperationOutputConsumers (TFOutput oper_out, TFInput* consumers, int max_consumers);
		// TODO:

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
					ret [i] = new TFOperation (arr [i], owns: false);
				return ret;
			}
		}

		// extern TF_AttrMetadata TF_OperationGetAttrMetadata (TF_Operation *oper, const char *attr_name, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TFAttributeMetadata TF_OperationGetAttrMetadata (TF_Operation oper, string attr_name, TF_Status status);

		public TFAttributeMetadata GetAttributeMetadata (string attrName, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var x = TF_OperationGetAttrMetadata (handle, attrName, cstatus.handle);
			cstatus.MaybeRaise (status);
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

		// extern TF_Operation * TF_GraphOperationByName (TF_Graph *graph, const char *oper_name);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_GraphOperationByName (TF_Graph graph, string oper_name);
		// TODO:

		// extern TF_Operation * TF_GraphNextOperation (TF_Graph *graph, size_t *pos);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_GraphNextOperation (TF_Graph graph, size_t* pos);
		// TODO:

		// extern void TF_OperationToNodeDef (TF_Operation *oper, TF_Buffer *output_node_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_OperationToNodeDef (TF_Operation oper, LLBuffer* output_node_def, TF_Status status);
		// TODO:
	}

	public class TFImportGraphDefOptions : TFDisposable {
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
			TF_ImportGraphDefOptionsSetPrefix (handle, prefix);
		}


	}

	public class TFSession : TFDisposable {
		// extern TF_Session * TF_NewSession (TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Session TF_NewSession (TF_Graph graph, TF_SessionOptions opts, TF_Status status);

		TFSession (IntPtr handle) : base (handle) { }

		public TFSession Create (TFGraph graph, TFSessionOptions sessionOptions, TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			var h = TF_NewSession (graph.handle, sessionOptions.handle, cstatus.handle);
			if (cstatus.MaybeRaise (status))
				return new TFSession (h);
			return null;
		}

		// extern TF_Session * TF_LoadSessionFromSavedModel (const TF_SessionOptions *session_options, const TF_Buffer *run_options, const char *export_dir, const char *const *tags, int tags_len, TF_Graph *graph, TF_Buffer *meta_graph_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Session TF_LoadSessionFromSavedModel (TF_SessionOptions session_options, LLBuffer* run_options, string export_dir, string []tags, int tags_len, TF_Graph graph, LLBuffer* meta_graph_def, TF_Status status);

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

				if (cstatus.MaybeRaise (status))
					return new TFSession (h);
			}
			return null;
		}

		// extern void TF_CloseSession (TF_Session *, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_CloseSession (TF_Session session, TF_Status status);

		public void CloseSession (TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			TF_CloseSession (handle, cstatus.handle);
			cstatus.MaybeRaise (status);
		}

		// extern void TF_DeleteSession (TF_Session *, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteSession (TF_Session session, TF_Status status);

		public void DeleteSession (TFStatus status = null)
		{
			var cstatus = TFStatus.Setup (status);
			TF_DeleteSession (handle, cstatus.handle);
			cstatus.MaybeRaise (status);
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

		public void Run (TFBuffer runOptions, TFOutput [] inputs, TFTensor [] inputValues, TFOutput [] outputs, TFTensor []outputValues, TFOperation [] targetOpers, TFBuffer runMetadata, TFStatus status = null)
		{
			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (inputValues == null)
				throw new ArgumentNullException (nameof (inputValues));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			if (outputValues == null)
				throw new ArgumentNullException (nameof (outputValues));
			if (targetOpers == null)
				throw new ArgumentNullException (nameof (targetOpers));
			int iLen = inputs.Length;
			if (iLen != inputValues.Length)
				throw new ArgumentException ("inputs and inputValues have different lengths", "inputs");
			int oLen = outputs.Length;
			if (oLen != outputValues.Length)
				throw new ArgumentException ("outputs and outputValues have different lengths", "outputs");

			// runOptions and runMetadata might be null
			var cstatus = TFStatus.Setup (status);

			// Create arrays for the unmanaged versions
			var ivals = new IntPtr [iLen];
			for (int i = 0; i < iLen; i++)
				ivals [i] = inputValues [i].handle;
			var ovals = new IntPtr [oLen];
			for (int i = 0; i < oLen; i++)
				ovals [i] = outputValues [i].handle;
			int tLen = targetOpers.Length;
			var topers = new IntPtr [tLen];
			for (int i = 0; i < tLen; i++)
				topers [i] = targetOpers [i].handle;

			unsafe
			{
				TF_SessionRun (handle, runOptions == null ? null : runOptions.LLBuffer, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, runMetadata == null ? null : runMetadata.LLBuffer, cstatus.handle);
			}
			cstatus.MaybeRaise (status);
		}

		// extern void TF_SessionPRunSetup (TF_Session, const TF_Output *inputs, int ninputs, const TF_Output *outputs, int noutputs, const TF_Operation *const *target_opers, int ntargets, const char **handle, TF_Status *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SessionPRunSetup (TF_Session session, TFOutput []inputs, int ninputs, TFOutput [] outputs, int noutputs, TF_Operation [] target_opers, int ntargets, out IntPtr returnHandle, TF_Status status);

		public struct PartialRunToken
		{
			internal IntPtr token;
		}

		public PartialRunToken PartialRunSetup (TFOutput [] inputs, TFOutput [] outputs, TFOperation [] targetOpers, TFStatus status = null)
		{
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
			cstatus.MaybeRaise (status);
			return new PartialRunToken () { token = returnHandle };
		}

		// extern void TF_SessionPRun (TF_Session *, const char *handle, const TF_Output *inputs, TF_Tensor *const *input_values, int ninputs, const TF_Output *outputs, TF_Tensor **output_values, int noutputs, const TF_Operation *const *target_opers, int ntargets, TF_Status *);
		static extern unsafe void TF_SessionPRun (TF_Session session, IntPtr partialHandle, TFOutput [] inputs, TF_Tensor [] input_values, int ninputs, TFOutput [] outputs, TF_Tensor [] output_values, int noutputs, TF_Operation [] target_opers, int ntargets, TF_Status status);
		public void PartialRun (PartialRunToken token, TFOutput [] inputs, TFTensor [] inputValues, TFOutput [] outputs, TFTensor [] outputValues, TFOperation [] targetOpers, TFStatus status = null)
		{
			if (inputs == null)
				throw new ArgumentNullException (nameof (inputs));
			if (inputValues == null)
				throw new ArgumentNullException (nameof (inputValues));
			if (outputs == null)
				throw new ArgumentNullException (nameof (outputs));
			if (outputValues == null)
				throw new ArgumentNullException (nameof (outputValues));
			if (targetOpers == null)
				throw new ArgumentNullException (nameof (targetOpers));
			int iLen = inputs.Length;
			if (iLen != inputValues.Length)
				throw new ArgumentException ("inputs and inputValues have different lengths", "inputs");
			int oLen = outputs.Length;
			if (oLen != outputValues.Length)
				throw new ArgumentException ("outputs and outputValues have different lengths", "outputs");

			// runOptions and runMetadata might be null
			var cstatus = TFStatus.Setup (status);

			// Create arrays for the unmanaged versions
			var ivals = new IntPtr [iLen];
			for (int i = 0; i < iLen; i++)
				ivals [i] = inputValues [i].handle;
			var ovals = new IntPtr [oLen];
			for (int i = 0; i < oLen; i++)
				ovals [i] = outputValues [i].handle;
			int tLen = targetOpers.Length;
			var topers = new IntPtr [tLen];
			for (int i = 0; i < tLen; i++)
				topers [i] = targetOpers [i].handle;

			unsafe
			{
				TF_SessionPRun (handle, token.token, inputs, ivals, iLen, outputs, ovals, oLen, topers, tLen, cstatus.handle);
			}
			cstatus.MaybeRaise (status);
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
			cstatus.MaybeRaise (status);
			return new TFLibrary (h);
		}

		// extern TF_Buffer TF_GetOpList (TF_Library *lib_handle);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TFBuffer TF_GetOpList (TF_Library lib_handle);
		// TODO:

		// extern void TF_DeleteLibraryHandle (TF_Library *lib_handle);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteLibraryHandle (TF_Library lib_handle);

		internal override void NativeDispose (TF_Status handle)
		{
			TF_DeleteLibraryHandle (handle);
		}

		// extern TF_Buffer * TF_GetAllOpList ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer* TF_GetAllOpList ();
		// TODO:
	}

	public enum TFDataType : uint
	{
		Float = 1,
		Double = 2,
		Int32 = 3,
		UInt8 = 4,
		Int16 = 5,
		Int8 = 6,
		String = 7,
		Complex64 = 8,
		Complex = 8,
		Int64 = 9,
		Bool = 10,
		QInt8 = 11,
		QUInt8 = 12,
		QInt32 = 13,
		BFloat16 = 14,
		QInt16 = 15,
		QUInt16 = 16,
		UInt16 = 17,
		Complex128 = 18,
		Half = 19,
		Resource = 20
	}

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

	[StructLayout (LayoutKind.Sequential)]
	public struct TFOutput
	{
		public unsafe TF_Operation Operation;
		public int Index;

		// extern int TF_OperationOutputNumConsumers (TF_Output oper_out);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern int TF_OperationOutputNumConsumers (TFOutput oper_out);

		public int NumConsumers => TF_OperationOutputNumConsumers (this);

		// extern TF_DataType TF_OperationOutputType (TF_Output oper_out);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern TFDataType TF_OperationOutputType (TFOutput oper_out);

		public TFDataType OutputType => TF_OperationOutputType (this);

	}

	public enum TFAttributeType : uint
	{
		String = 0,
		Int = 1,
		Float = 2,
		Bool = 3,
		Type = 4,
		Shape = 5,
		Tensor = 6,
		Placeholder = 7,
		Func = 8
	}

	[StructLayout (LayoutKind.Sequential)]
	public struct TFAttributeMetadata
	{
		public byte IsList;
		public long ListSize;
		public TFAttributeType Type;
		public long TotalSize;
	}

}
