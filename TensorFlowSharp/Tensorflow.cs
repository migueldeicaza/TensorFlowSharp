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

using size_t = System.UIntPtr;
using System.Numerics;
using System.Collections.Generic;

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

		// extern TF_Buffer * TF_GetAllOpList ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_GetAllOpList ();

		public static TFBuffer GetAllOpList ()
		{
			return new TFBuffer (TF_GetAllOpList ());
		}
	}

	public abstract class TFDisposable : IDisposable
	{
		internal IntPtr handle;
		public IntPtr Handle => handle;

		public TFDisposable ()
		{ }

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

		// Must be implemented in subclasses to dispose the unmanaged object, it does
		// not need to take care of zeroing out the handle, that is done by the Dispose
		// method inherited from TFDisposable
		internal abstract void NativeDispose (IntPtr handle);

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

	public class TFException : Exception {
		public TFException (string message) : base (message) { }
	}

	/// <summary>
	/// Used to track the result of TensorFlow operations.
	/// </summary>
	/// <remarks>
	/// TFStatus is used to track the status of a call to some TensorFlow
	/// operations.   Instances of this object are passed to various
	/// TensorFlow operations and you can use the <see cref="P:TensorFlow.TFStatus.Ok"/>
	/// to quickly check if the operation succeeded, or get more detail from the
	/// <see cref="P:TensorFlow.TFStatus.StatusCode"/> and a human-readable text
	/// using the <see cref="P:TensorFlow.TFStatus.StatusMessage"/> property.
	/// 
	/// The convenience <see cref="M:TensorFlow.TFStatus.Raise"/> can be used
	/// to raise a <see cref="P:TensorFlow.TFException"/> if the status of the
	/// operation did not succeed.
	/// </remarks>
	public class TFStatus : TFDisposable
	{
		// extern TF_Status * TF_NewStatus ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe TF_Status TF_NewStatus ();

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

	[StructLayout (LayoutKind.Sequential)]
	internal struct LLBuffer
	{
		internal IntPtr data;
		internal size_t length;
		internal IntPtr data_deallocator;
	}	

	/// <summary>
	/// Holds a block of data.
	/// </summary>
	/// <remarks>
	/// Use the TFBuffer to blobs of data into TensorFlow, or to retrieve blocks
	/// of data out of TensorFlow.
	/// 
	/// There are two constructors to wrap existing data, one to wrap blocks that are 
	/// pointed to by an IntPtr and one that takes a byte array that we want to wrap.
	/// 
	/// The empty constructor can be used to create a new TFBuffer that can be populated
	/// by the TensorFlow library and returned to user code.
	/// 
	/// Typically, the data consists of a serialized protocol buffer, but other data
	/// may also be held in a buffer.
	/// </remarks>
	// TODO: the string ctor
	// TODO: perhaps we should have an implicit byte [] conversion that just calls ToArray?
	public class TFBuffer : TFDisposable
	{
		// extern TF_Buffer * TF_NewBufferFromString (const void *proto, size_t proto_len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer* TF_NewBufferFromString (IntPtr proto, IntPtr proto_len);

		// extern TF_Buffer * TF_NewBuffer ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer* TF_NewBuffer ();

		internal TFBuffer (IntPtr handle) : base (handle) { }

		unsafe public TFBuffer () : base ((IntPtr)TF_NewBuffer ())
		{
		}

		/// <summary>
		/// Signature of the method that is invoked to release the data.  
		/// </summary>
		/// <remarks>
		/// Methods of this signature are invoked with the data pointer and the
		/// lenght pointer when then TFBuffer no longer needs to hold on to the
		/// data.
		/// </remarks>
		public delegate void BufferReleaseFunc (IntPtr data, IntPtr lenght);

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by wrapping the unmanaged resource pointed by the buffer.
		/// </summary>
		/// <param name="buffer">Pointer to the data that will be wrapped.</param>
		/// <param name="size">The size of the buffer to wrap.</param>
		/// <param name="release">Optional, if not null, this method will be invoked to release the block.</param>
		/// <remarks>
		/// This constructor wraps the buffer as a the data to be held by the <see cref="T:TensorFlow.TFBuffer"/>,
		/// if the release parameter is null, then you must ensure that the data is not released before the TFBuffer
		/// is no longer in use.   If the value is not null, the provided method will be invoked to release
		/// the data when the TFBuffer is disposed, or the contents of the buffer replaced.
		/// </remarks>
		unsafe public TFBuffer (IntPtr buffer, long size, BufferReleaseFunc release) : base ((IntPtr)TF_NewBuffer ())
		{
			LLBuffer* buf = (LLBuffer*)handle;
			buf->data = buffer;
			buf->length = (size_t)size;
			if (release == null)
				buf->data_deallocator = IntPtr.Zero;
			else
				buf->data_deallocator = Marshal.GetFunctionPointerForDelegate (release);
		}

		internal static void FreeBlock (IntPtr data, IntPtr lenght)
		{
			Marshal.FreeHGlobal (data);
		}

		static IntPtr FreeBufferFunc;

		static TFBuffer ()
		{
			FreeBufferFunc = Marshal.GetFunctionPointerForDelegate<BufferReleaseFunc> (FreeBlock);
		}


		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by making a copy of the provided byte array.
		/// </summary>
		/// <param name="buffer">Buffer of data that will be wrapped.</param>
		/// <remarks>
		/// This constructor makes a copy of the data into an unmanaged buffer, 
		/// so the byte array is not pinned.
		/// </remarks>
		public TFBuffer (byte [] buffer) : this (buffer, 0, buffer.Length) { }

		/// <summary>
		/// Initializes a new instance of the <see cref="T:TensorFlow.TFBuffer"/> by making a copy of the provided byte array.
		/// </summary>
		/// <param name="buffer">Buffer of data that will be wrapped.</param>
		/// <param name="start">Starting offset into the buffer to wrap.</param>
		/// <param name="count">Number of bytes from the buffer to keep.</param>
		/// <remarks>
		/// This constructor makes a copy of the data into an unmanaged buffer, 
		/// so the byte array is not pinned.
		/// </remarks>
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

		internal override void NativeDispose (IntPtr handle)
		{
			unsafe { TF_DeleteBuffer ((LLBuffer*)handle); }
		}

		// extern TF_Buffer TF_GetBuffer (TF_Buffer *buffer);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe LLBuffer TF_GetBuffer (LLBuffer *buffer);

		/// <summary>
		/// Returns a byte array representing the data wrapped by this buffer.
		/// </summary>
		/// <returns>The array.</returns>
		public byte [] ToArray ()
		{
			if (handle == IntPtr.Zero)
				return null;
			
			unsafe
			{
				var lb = (LLBuffer*)handle;

				var result = new byte [(int) lb->length];
				Marshal.Copy (lb->data, result, 0, (int) lb->length);

				return result;
			}
		}
	}

	/// <summary>
	/// TFTensor holds a multi-dimensional array of elements of a single data type.
	/// </summary>
	/// <remarks>
	/// You can create tensors with the various constructors in this class, or using
	/// the implicit conversions from various data types into a TFTensor.
	/// 
	/// The implicit conversions for basic types produce tensors of one dimesion with
	/// a single element, while the implicit conversion from an array, expects a multi-dimensional
	/// array that is converted into a tensor of the right dimensions.
	/// 
	/// The special "String" tensor data type that you will find in TensorFlow documentation
	/// really represents a byte array.   You can create string tensors by using the <see cref="M:TensorFlow.TFTensor.CreateString"/> 
	/// method that takes a byte array buffer as input.
	/// </remarks>
	public class TFTensor : TFDisposable
	{
		public delegate void Deallocator (IntPtr data, IntPtr size, IntPtr deallocatorData);

		// extern TF_Tensor * TF_NewTensor (TF_DataType, const int64_t *dims, int num_dims, void *data, size_t len, void (* deallocator)(void *, size_t, void *), void *deallocator_arg);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_NewTensor (TFDataType dataType, long [] dims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg);

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_NewTensor (TFDataType dataType, IntPtr zeroDims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg);

		internal TFTensor (IntPtr handle) : base (handle) { }
		internal static void FreeTensorData (IntPtr data, IntPtr len, IntPtr closure)
		{
			Marshal.FreeHGlobal (data);
		}

		internal static void FreeTensorHandle (IntPtr data, IntPtr len, IntPtr closure)
		{
			var gch = GCHandle.FromIntPtr (closure);
			gch.Free ();
		}

		// TODO: Other overloads we could add: String, Complex (float), Bool, QInt8, QUInt8, QInt32, Bfloat16,
		// QInt16, QUint16, Half, Resource
		// TODO: not clear that this is very useful (the dims versions), perhaps to reduce the surface of
		// construcors these rarer blobs should be "FromSpec" or something like that
		public TFTensor (long [] dims, sbyte [] data, int start, int count)   : base (SetupTensor (TFDataType.Int8, dims, data, start, count, size: 2)) { }
		public TFTensor (long [] dims, byte [] data, int start, int count)    : base (SetupTensor (TFDataType.UInt8, dims, data, start, count, size: 1)) { }
		public TFTensor (long [] dims, short [] data, int start, int count)   : base (SetupTensor (TFDataType.Int16, dims, data, start, count, size: 2)) { }
		public TFTensor (long [] dims, ushort [] data, int start, int count)  : base (SetupTensor (TFDataType.UInt16, dims, data, start, count, size: 2)) { }
		public TFTensor (long [] dims, int [] data, int start, int count)     : base (SetupTensor (TFDataType.Int32, dims, data, start, count, size: 4)) { }
		public TFTensor (long [] dims, float [] data, int start, int count)   : base (SetupTensor (TFDataType.Float, dims, data, start, count, size: 4)) { }
		public TFTensor (long [] dims, double [] data, int start, int count)  : base (SetupTensor (TFDataType.Double, dims, data, start, count, size: 8)) { }
		public TFTensor (long [] dims, long [] data, int start, int count)    : base (SetupTensor (TFDataType.Int64, dims, data, start, count, size: 8)) { }
		public TFTensor (long [] dims, Complex [] data, int start, int count) : base (SetupTensor (TFDataType.Complex128, dims, data, start, count, size: 16)) { }
		public TFTensor (long [] dims, sbyte [] data) : base (SetupTensor (TFDataType.Int8, dims, data, size: 2)) { }
		public TFTensor (long [] dims, byte [] data) : base (SetupTensor (TFDataType.UInt8, dims, data, size: 1)) { }
		public TFTensor (long [] dims, short [] data) : base (SetupTensor (TFDataType.Int16, dims, data, size: 2)) { }
		public TFTensor (long [] dims, ushort [] data) : base (SetupTensor (TFDataType.UInt16, dims, data, size: 2)) { }
		public TFTensor (long [] dims, int [] data) : base (SetupTensor (TFDataType.Int32, dims, data, size: 4)) { }
		public TFTensor (long [] dims, float [] data) : base (SetupTensor (TFDataType.Float, dims, data, size: 4)) { }
		public TFTensor (long [] dims, double [] data) : base (SetupTensor (TFDataType.Double, dims, data, size: 8)) { }
		public TFTensor (long [] dims, long [] data) : base (SetupTensor (TFDataType.Int64, dims, data, size: 8)) { }
		public TFTensor (long [] dims, Complex [] data) : base (SetupTensor (TFDataType.Complex128, dims, data, size: 16)) { }

		public unsafe TFTensor (int value)
		{
			var v = (int*)Marshal.AllocHGlobal (sizeof (int));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int32, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (int), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (sbyte value)
		{
			var v = (sbyte*)Marshal.AllocHGlobal (sizeof (sbyte));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int8, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (sbyte), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (short value)
		{
			var v = (short*)Marshal.AllocHGlobal (sizeof (short));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int16, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (short), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (ushort value)
		{
			var v = (ushort*)Marshal.AllocHGlobal (sizeof (ushort));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int16, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (ushort), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (byte value)
		{
			var v = (int*)Marshal.AllocHGlobal (sizeof (byte));
			*v = value;
			handle = TF_NewTensor (TFDataType.UInt8, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (byte), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (Complex value)
		{
			var v = (Complex*)Marshal.AllocHGlobal (sizeof (Complex));
			*v = value;
			handle = TF_NewTensor (TFDataType.Complex128, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (Complex), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (float value)
		{
			var v = (float*)Marshal.AllocHGlobal (sizeof (float));
			*v = value;
			handle = TF_NewTensor (TFDataType.Float, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (float), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		public unsafe TFTensor (double value)
		{
			var v = (double*)Marshal.AllocHGlobal (sizeof (double));
			*v = value;
			handle = TF_NewTensor (TFDataType.Double, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (double), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}
		public unsafe TFTensor (long value)
		{
			var v = (long*)Marshal.AllocHGlobal (sizeof (long));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int64, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (long), deallocator: FreeTensorData, deallocator_arg: IntPtr.Zero);
		}

		// Convenience, should I add T[,] and T[,,] as more convenience ones?
		public TFTensor (sbyte [] data) : base (SetupTensor (TFDataType.Int8, data, size: 2)) { }
		public TFTensor (byte [] data) : base (SetupTensor (TFDataType.UInt8, data, size: 1)) { }
		public TFTensor (short [] data) : base (SetupTensor (TFDataType.Int16, data, size: 2)) { }
		public TFTensor (ushort [] data) : base (SetupTensor (TFDataType.UInt16, data, size: 2)) { }
		public TFTensor (int [] data) : base (SetupTensor (TFDataType.Int32, data, size: 4)) { }
		public TFTensor (float [] data) : base (SetupTensor (TFDataType.Float, data, size: 4)) { }
		public TFTensor (double [] data) : base (SetupTensor (TFDataType.Double, data, size: 8)) { }
		public TFTensor (long [] data) : base (SetupTensor (TFDataType.Int64, data, size: 8)) { }
		public TFTensor (Complex [] data) : base (SetupTensor (TFDataType.Complex128, data, size: 16)) { }

		public unsafe static TFTensor CreateString (byte [] buffer)
		{
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));
			//
			// TF_STRING tensors are encoded with a table of 8-byte offsets followed by
			// TF_StringEncode-encoded bytes.
			//
			var size = TFString.TF_StringEncodedSize ((UIntPtr) buffer.Length);
			IntPtr handle = TF_AllocateTensor (TFDataType.String, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

			// Clear offset table
			IntPtr dst = TF_TensorData (handle);
			Marshal.WriteInt64 (dst, 0);
			var status = TFStatus.TF_NewStatus ();
			fixed (byte *src = &buffer [0])
			{
				TFString.TF_StringEncode (src, (UIntPtr) buffer.Length, (sbyte *)(dst + 8), size, status);
				var ok = TFStatus.TF_GetCode (status) == TFCode.Ok;
				TFStatus.TF_DeleteStatus (status);
				if (!ok)
					return null;
			}
			return new TFTensor (handle);
		}

		// Convenience function to factor out the setup of a new tensor from an array
		static IntPtr SetupTensor (TFDataType dt, long [] dims, Array data, int size)
		{
			return SetupTensor (dt, dims, data, start: 0, count: data.Length, size: size);
		}

		// Convenience function to factor out the setup of a new tensor from an array
		static IntPtr SetupTensor (TFDataType dt, Array data, int size)
		{
			long [] dims = new long [data.Rank];
			for (int i = 0; i < dims.Length; i++)
				dims [i] = data.GetLength (i);
			
			return SetupTensor (dt, dims, data, start: 0, count: data.Length, size: size);
		}

		// Use for single dimension arrays 
		static IntPtr SetupTensor (TFDataType dt, long [] dims, Array data, int start, int count, int size)
		{
			if (start < 0 || start > data.Length - count)
				throw new ArgumentException ("start + count > Array size");
			
			var dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned);

			if (dims == null)
				return TF_NewTensor (dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject ()+ start*size, (UIntPtr)(count*size), FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
			else
				return TF_NewTensor (dt, dims, dims.Length, dataHandle.AddrOfPinnedObject () + start * size, (UIntPtr)(count*size), FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
		}

		// Use for multiple dimension arrays 
		static IntPtr SetupMulti (TFDataType dt, long [] dims, Array data, long bytes)
		{
			var dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned);

			if (dims == null)
				return TF_NewTensor (dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject (), (UIntPtr)bytes, FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
			else
				return TF_NewTensor (dt, dims, dims.Length, dataHandle.AddrOfPinnedObject (), (UIntPtr)bytes, FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
		}

		// 
		// Factory methods to create tensors from a constant
		//
		// TODO: add more data types

		public static implicit operator TFTensor (int value)
		{
			return new TFTensor (value);
		}

		public static implicit operator TFTensor (long value)
		{
			return new TFTensor (value);
		}

		unsafe public static implicit operator TFTensor (double value)
		{
			return new TFTensor (value);
		}

		unsafe public static implicit operator TFTensor (float value)
		{
			return new TFTensor (value);
		}

		unsafe public static implicit operator TFTensor (Complex value)
		{
			return new TFTensor (value);
		}

		unsafe public static implicit operator TFTensor (byte value)
		{
			return new TFTensor (value);
		}

		unsafe public static implicit operator TFTensor (Array array)
		{
			if (array == null)
				throw new ArgumentNullException (nameof (array));
			// TODO: ensure that we do not have arrays of arrays.
			var t = array.GetType ().GetElementType ();
			var tc = Type.GetTypeCode (t);
			TFDataType dt;
			long size = 0;
			switch (tc) {
			case TypeCode.Boolean:
				dt = TFDataType.Bool;
				size = 1;
				break;
			case TypeCode.SByte:
				dt = TFDataType.Int8;
				size = 1;
				break;
			case TypeCode.Byte:
				dt = TFDataType.UInt8;
				size = 1;
				break;
			case TypeCode.Int16:
				dt = TFDataType.Int16;
				size = 2;
				break;
			case TypeCode.UInt16:
				dt = TFDataType.UInt16;
				size = 2;
				break;
			case TypeCode.Int32:
				dt = TFDataType.Int32;
				size = 4;
				break;
			case TypeCode.Int64:
				dt = TFDataType.Int64;
				size = 8;
				break;
			case TypeCode.Single:
				dt = TFDataType.Float;
				size = 4;
				break;
			case TypeCode.Double:
				dt = TFDataType.Double;
				size = 8;
				break;
			default:
				// Check types that are not handled by the typecode
				if (t is Complex){
					size = 16;
					dt = TFDataType.Complex128;
				} else
					throw new ArgumentException ($"The data type {t} is not supported");
			}

			var dims = new long [array.Rank];
			for (int i = 0; i < array.Rank; i++) {
				dims [i] = array.GetLength (i);
				size *= (int) dims [i];
			}
			var newTensor = new TFTensor (SetupMulti (dt, dims, array, size));
			return newTensor;
		}

		// General purpose constructor, specifies data type and gets pointer to buffer
		// Is the default good, one where we let the user provide their own deallocator, or should we make a copy in that case?
		public TFTensor (TFDataType dataType, long [] dims, IntPtr data, size_t dataSize, Deallocator deallocator, IntPtr deallocatorData) : base (IntPtr.Zero)
		{
			if (dims == null)
				throw new ArgumentNullException ("dims");

			handle = TF_NewTensor (dataType, dims, dims.Length, data, dataSize, deallocator, deallocatorData);

		}

		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteTensor (handle);
		}

		// extern TF_Tensor * TF_AllocateTensor (TF_DataType, const int64_t *dims, int num_dims, size_t len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_AllocateTensor (TFDataType dataType, long [] dims, int num_dims, size_t len);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_AllocateTensor (TFDataType dataType, IntPtr zeroDim, int num_dims, size_t len);

		public TFTensor (TFDataType dataType, long [] dims, int size) : base (IntPtr.Zero)
		{
			if (dims == null)
				throw new ArgumentNullException ("dims");
			handle = TF_AllocateTensor (dataType, dims, dims.Length, (size_t)size);
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

		/// <summary>
		/// Returns the number of dimensions in the tensor.
		/// </summary>
		/// <remarks>
		/// For single-dimension tensors the return is 1, 2 dimensions is 2 and so on.
		/// </remarks>
		public int NumDims => TF_NumDims (handle);

		// extern int64_t TF_Dim (const TF_Tensor *tensor, int dim_index);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe long TF_Dim (TF_Tensor tensor, int dim_index);

		/// <summary>
		/// Returns the number of elements on a specific dimension in the tensor.
		/// </summary>
		/// <returns>The tensor dimension.</returns>
		/// <param name="dimIndex">Dimension that you are querying.</param>
		/// <remarks>
		/// If you have a tensor of 3 elements by 5, represented by [3 5],
		/// the GetTensorDimension(0) will return 3, the GetTensorDimension(1)
		/// will return 5.
		/// </remarks>
		public long GetTensorDimension (int dimIndex)
		{
			return TF_Dim (handle, dimIndex);
		}

		// extern size_t TF_TensorByteSize (const TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe size_t TF_TensorByteSize (TF_Tensor tensor);

		public size_t TensorByteSize => TF_TensorByteSize (handle);

		// extern void * TF_TensorData (const TF_Tensor *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe IntPtr TF_TensorData (TF_Tensor tensor);

		/// <summary>
		/// Returns a pointer to the raw data in the tensor.
		/// </summary>
		/// <remarks>
		/// The contents of the Data must be interpreted according to the type of the
		/// data as described by the DataType property.   The amount of data
		/// is given by the the TensorByteSize property.
		/// </remarks>
		public IntPtr Data => TF_TensorData (handle);

		/// <summary>
		/// Returns the tensor shape, this is an array whose size determines the number of dimensions on the tensor, and each element is the size of the dimension
		/// </summary>
		/// <remarks>
		///     An array of size 0 is used for constants, an array of size 1 is used
		///     for single-dimension arrays, where the dimension is the value of the
		///     first element.   And so on.
		/// </remarks>
		public long [] Shape {
			get {
				var dims = new long [TF_NumDims (handle)];
				for (int i = 0; i < dims.Length; i++) 
					dims [i] = (int) TF_Dim (handle, i);

				return dims;
			}
		}

		static Type TypeFromTensorType (TFDataType type)
		{
			switch (type) {
			case TFDataType.Float:
				return typeof (float);
			case TFDataType.Double:
				return typeof (double);
			case TFDataType.Int32:
				return typeof (int);
			case TFDataType.UInt8:
				return typeof (byte);
			case TFDataType.Int16:
				return typeof (short);
			case TFDataType.Int8:
				return typeof (sbyte);
			case TFDataType.String:
				return typeof (TFString);
			case TFDataType.Int64:
				return typeof (long);
			case TFDataType.Bool:
				return typeof (bool);
			case TFDataType.UInt16:
				return typeof (ushort);
			case TFDataType.Complex128:
				return typeof (Complex);
			default:
				return null;
			}
		}

		static unsafe object FetchSimple (TFDataType dt, IntPtr data)
		{
			switch (dt) {
			case TFDataType.Float:
				return *(float*)data;
			case TFDataType.Double:
				return *(double*)data;
			case TFDataType.Int32:
				return *(int*)data;
			case TFDataType.UInt8:
				return *(byte*)data;
			case TFDataType.Int16:
				return *(short*)data;
			case TFDataType.Int8:
				return *(sbyte*)data;
			case TFDataType.String:
				throw new NotImplementedException ();
			case TFDataType.Int64:
				return *(long*)data;
			case TFDataType.Bool:
				return *(bool*)data;
			case TFDataType.UInt16:
				return *(ushort*)data;
			case TFDataType.Complex128:
				return *(Complex*)data;
			default:
				return null;
			}	
		}

		unsafe static void Copy (IntPtr src, void* target, int size)
		{
			Buffer.MemoryCopy ((void*)src, target, size, size);
		}

		static unsafe void FetchFlatArray (Array target, TFDataType dt, IntPtr data)
		{
			int len = target.Length;
			switch (dt) {
			case TFDataType.Int8:
				var asbyte = (sbyte [])target;
				fixed (sbyte* p = &asbyte [0])
					Copy (data, p, len);
				return;
			case TFDataType.Bool:
				var abool = (bool [])target;
				fixed (bool* p = &abool [0])
					Copy (data, p, len);
				return;
			case TFDataType.UInt16:
				var aushort = (ushort [])target;
				fixed (ushort* p = &aushort [0])
					Copy (data, p, len * 2);
				return;
			case TFDataType.Complex128:
				var acomplex = (Complex [])target;
				fixed (Complex* p = &acomplex [0])
					Copy (data, p, len * sizeof (Complex));
				return;
			case TFDataType.Float:
				var afloat = (float [])target;
				fixed (float* p = &afloat [0])
					Copy (data, p, len * sizeof(float));
				return;
			case TFDataType.Double:
				var adouble = (double [])target;
				fixed (double* p = &adouble [0])
					Copy (data, p, len * sizeof (double));
				return;
			case TFDataType.Int32:
				var aint = (int [])target;
				fixed (int* p = &aint [0])
					Copy (data, p, len * sizeof (double));
				return;
			case TFDataType.UInt8:
				var abyte = (byte [])target;
				fixed (byte* p = &abyte [0])
					Copy (data, p, len * sizeof (byte));
				return;
			case TFDataType.Int16:
				var ashort = (short [])target;
				fixed (short* p = &ashort [0])
					Copy (data, p, len * sizeof (short));
				return;
			case TFDataType.Int64:
				var along = (long [])target;
				fixed (long* p = &along [0])
					Copy (data, p, len * sizeof (long));
				return;
			case TFDataType.String:
				// need to return an array of TFStrings []
				throw new NotImplementedException ();
			default:
				throw new NotImplementedException ();
			}
		}

		static unsafe object FetchJaggedArray (Type t, TFDataType dt, ref IntPtr data, long [] shape, int level = 0)
		{
			Array target;

			// If we are at the last node
			if (level == shape.Length - 1) {
				target = Array.CreateInstance (t, shape [level]);

				for (long l = 0; l < shape [level]; l++)
				switch (dt) {
				case TFDataType.Float:
					target.SetValue ((*(float*)data), l);
					data += 4;
					break;
				case TFDataType.Double:
					target.SetValue ((*(double*)data), l);
					data += 8;
					break;
				case TFDataType.Int32:
					target.SetValue ((*(int*)data), l);
					data += 4;
					break;
				case TFDataType.UInt8:
					target.SetValue ((*(byte*)data), l);
					data += 1;
					break;
				case TFDataType.Int16:
					target.SetValue ((*(short*)data), l);
					data += 2;
					break;
				case TFDataType.Int8:
					target.SetValue ((*(sbyte*)data), l);
					data += 1;
					break;
				case TFDataType.Int64:
					target.SetValue ((*(long*)data), l);
					data += 8;
					break;
				case TFDataType.Bool:
					target.SetValue ((*(bool*)data), l);
					data += 1;
					break;
				case TFDataType.Complex128:
					target.SetValue ((*(Complex*)data), l);
					data += sizeof (Complex);
					break;
				case TFDataType.String:
					throw new NotImplementedException ("String decoding not implemented for tensor vecotrs yet");
				default:
					throw new NotImplementedException ();
				}				
			} else {
				target = null;

				long top = shape [level];
				if (top < Int32.MaxValue) {
					int itop = (int)top;

					for (int i = 0; i < itop; i++) {
						var childArray = FetchJaggedArray (t, dt, ref data, shape, level + 1);
						if (target == null) 
							target = Array.CreateInstance (childArray.GetType (), shape [level]);
						
						target.SetValue (childArray, i);
					}
				} else {
					for (int l = 0; l < top; l++){

						var chidArray = FetchJaggedArray (t, dt, ref data, shape, level + 1);
						if (target == null) 
							target = Array.CreateInstance (chidArray.GetType (), shape [level]);
						
						target.SetValue (chidArray, l);
					}
				}
				return target;
			}

			return target;
		}

                static void FetchMultiDimensionalArray (Array target, TFDataType dt, IntPtr data, long [] shape)
		{
			var idx = new int [shape.Length];
			for (int i = 0; i < shape.Length; i++) {
				if (shape [i] > Int32.MaxValue)
					throw new ArgumentOutOfRangeException ("Shape can not be longer than 32 bits");
				idx [i] = (int) shape [i];
			}
			Copy (target, dt, shape, idx, shape.Length - 1, ref data);
		}

		static unsafe void Copy (Array target, TFDataType dt, long [] shape, int [] idx, int level, ref IntPtr data)
		{
			if (level > 0) {
				for (idx [level] = 0; idx [level] < shape [level]; idx [level]++)
					Copy (target, dt, shape, idx, level - 1, ref data);
			} else {
				for (idx [0] = 0; idx [0] < shape [0]; idx [0]++){
					switch (dt) {
					case TFDataType.Float:
						target.SetValue ((*(float *) data), idx);
						data += 4;
						break;
					case TFDataType.Double:
						target.SetValue ((*(double*)data), idx);
						data += 8;
						break;
					case TFDataType.Int32:
						target.SetValue ((*(int*)data), idx);
						data += 4;
						break;
					case TFDataType.UInt8:
						target.SetValue ((*(byte*)data), idx);
						data += 1;
						break;
					case TFDataType.Int16:
						target.SetValue ((*(short*)data), idx);
						data += 2;
						break;
					case TFDataType.Int8:
						target.SetValue ((*(sbyte*)data), idx);
						data += 1;
						break;
					case TFDataType.Int64:
						target.SetValue ((*(long*)data), idx);
						data += 8;
						break;
					case TFDataType.Bool:
						target.SetValue ((*(bool*)data), idx);
						data += 1;
						break;
					case TFDataType.Complex128:
						target.SetValue ((*(Complex*)data), idx);
						data += sizeof(Complex);
						break;
					case TFDataType.String:
						throw new NotImplementedException ("String decoding not implemented for tensor vecotrs yet");
					default:
						throw new NotImplementedException ();
					}
				}
			}
		}

		/// <summary>
		/// Returns the value of the Tensor as a C# type if possible, or null if the data type can not be represented in C#
		/// </summary>
		/// <param name="jagged">
		/// The default is set to false, which returns .NET multi-dimensional arrays for multi-dimensional
		/// tensors.    This is useful to feed the data back as a TFTensor created from an array.   Set to
		/// true if you want to get arrays pointing to arrays, which are slightly more convenient to work
		/// with from C#
		/// </param>
		/// <remarks>
		/// Jagged arrays create various intermediate arrays, while multi-dimensional arrays are more
		/// efficient memory-wise.
		/// </remarks>
		/// <returns>The value encodes the contents of the tensor, and could include simple values, arrays and multi-dimensional values.</returns>
		public object GetValue (bool jagged = false)
		{
			var dims = NumDims;
			if (dims == 0) 
				return FetchSimple (TensorType, Data);
			
			var t = TypeFromTensorType (TensorType);
			if (t == null)
				return null;

			if (dims == 1){
				var result = Array.CreateInstance (t, Shape [0]);
				FetchFlatArray (result, TensorType, Data);
				return result;
			} else {
				if (jagged){
					IntPtr data = Data;
					return FetchJaggedArray (t, TensorType, ref data, Shape);
				} else {
					var result = Array.CreateInstance (t, Shape);
					FetchMultiDimensionalArray (result, TensorType, Data, Shape);
					return result;
				}
			}
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

	public class TFSessionOptions : TFDisposable
	{
		// extern TF_SessionOptions * TF_NewSessionOptions ();
		[DllImport (NativeBinding.TensorFlowLibrary)]
		internal static extern unsafe TF_SessionOptions TF_NewSessionOptions ();

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
		public void SetTarget (string target)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			
			TF_SetTarget (handle, target);
		}

		// extern void TF_SetConfig (TF_SessionOptions *options, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetConfig (TF_SessionOptions options, IntPtr proto, size_t proto_len, TF_Status status);


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

		// extern void TF_DeleteGraph (TF_Graph *);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteGraph (TF_Graph graph);
		internal override void NativeDispose (IntPtr handle)
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
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();

			var cstatus = TFStatus.Setup (status);
			if (dims == null)
				TF_GraphSetTensorShape (handle, output, IntPtr.Zero, 0, cstatus.handle);
			else
				TF_GraphSetTensorShape (handle, output, ref dims, dims.Length, cstatus.handle);
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
		static extern unsafe void TF_GraphGetTensorShape (TF_Graph graph, TFOutput output, ref long [] dims, int num_dims, TF_Status status);

		public void GetTensorShape (TFOutput output, long [] dims, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (dims == null)
				throw new ArgumentNullException ("dims");
			var cstatus = TFStatus.Setup (status);
			TF_GraphGetTensorShape (handle, output, ref dims, dims.Length, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
		}

		// extern void TF_GraphToGraphDef (TF_Graph *graph, TF_Buffer *output_graph_def, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_GraphToGraphDef (TF_Graph graph, LLBuffer* output_graph_def, TF_Status status);

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
		/// <param name="status">Status.</param>
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
			TF_GraphGetTensorShape (handle, output, ref ret, ndims, cstatus.handle);
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

		string MakeName (string operName, string userName)
		{
			userName = (userName == null) ? operName : userName;
			if (CurrentNameScope == "")
				return userName;
			return CurrentNameScope + "/" + userName;
		}
	}

	public class TFScope : IDisposable 
	{
		TFGraph container;
		string name;

		internal TFScope (TFGraph container)
		{
			this.container = container;
			name = container.CurrentNameScope;
		}

		public void Dispose ()
		{
			container.CurrentNameScope = name;
		}
	}

	/// <summary>
	/// Low-level TensorFlow operation builder
	/// </summary>
	/// <remarks>
	/// This is the low-level API that is used to create operations by manually specificying all
	/// the parameters of an operation (inputs, outputs, attribute descriptions) that can then
	/// be attached into a graph.
	/// 
	/// Generally, you will instead be using the methods surfaced in <see cref="T:TensorFlow.TFGraph"/> 
	/// that surfaces a C# high-level API that has already been bound to the built-in TensorFlow
	/// nodes.
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

		public void SetDevice (string device)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (device == null)
				throw new ArgumentNullException ("device");
			TF_SetDevice (handle, device);
		}

		// extern void TF_AddInput (TF_OperationDescription *desc, TF_Output input);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddInput (TF_OperationDescription desc, TFOutput input);

		public void AddInput (TFOutput input)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			TF_AddInput (handle, input);
		}

		// extern void TF_AddInputList (TF_OperationDescription *desc, const TF_Output *inputs, int num_inputs);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddInputList (TF_OperationDescription desc, TFOutput [] inputs, int num_inputs);

		public void AddInputs (params TFOutput [] inputs)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (inputs == null || inputs.Length == 0)
				return;

			TF_AddInputList (handle, inputs, inputs.Length);
		}

		// extern void TF_AddControlInput (TF_OperationDescription *desc, TF_Operation *input);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_AddControlInput (TF_OperationDescription desc, TF_Operation input);

		public void AddControlInput (TFOperation input)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (input == null)
				throw new ArgumentNullException ("input");

			TF_AddControlInput (handle, input.handle);
		}

		// extern void TF_ColocateWith (TF_OperationDescription *desc, TF_Operation *op);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_ColocateWith (TF_OperationDescription desc, TF_Operation op);

		public void ColocateWith (TFOperation op)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (op == null)
				throw new ArgumentNullException ("op");
			TF_ColocateWith (handle, op.handle);
		}

		// extern void TF_SetAttrString (TF_OperationDescription *desc, const char *attr_name, const void *value, size_t length);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrString (TF_OperationDescription desc, string attr_name, IntPtr value, size_t length);

		public void SetAttr (string attrName, string value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			var bytes = Encoding.UTF8.GetBytes (value);
			var buf = Marshal.AllocHGlobal (bytes.Length + 1);
			Marshal.Copy (bytes, 0, buf, bytes.Length);

			TF_SetAttrString (handle, attrName, buf, (UIntPtr)bytes.Length);
		}

		// extern void TF_SetAttrStringList (TF_OperationDescription *desc, const char *attr_name, const void *const *values, const size_t *lengths, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrStringList (TF_OperationDescription desc, string attr_name, IntPtr [] values, UIntPtr [] lengths, int num_values);
		public void SetAttr (string attrName, string [] values)
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
		}


		// extern void TF_SetAttrInt (TF_OperationDescription *desc, const char *attr_name, int64_t value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrInt (TF_OperationDescription desc, string attr_name, long value);

		public void SetAttr (string attrName, long value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrInt (handle, attrName, value);
		}

		// extern void TF_SetAttrIntList (TF_OperationDescription *desc, const char *attr_name, const int64_t *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrIntList (TF_OperationDescription desc, string attr_name, long [] values, int num_values);

		public void SetAttr (string attrName, long [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrIntList (handle, attrName, values, values.Length);
		}


		// extern void TF_SetAttrFloat (TF_OperationDescription *desc, const char *attr_name, float value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrFloat (TF_OperationDescription desc, string attr_name, float value);

		public void SetAttr (string attrName, float value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrFloat (handle, attrName, value);
		}

		// extern void TF_SetAttrFloatList (TF_OperationDescription *desc, const char *attr_name, const float *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrFloatList (TF_OperationDescription desc, string attr_name, float [] values, int num_values);

		public void SetAttr (string attrName, float [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrFloatList (handle, attrName, values, values.Length);
		}

		// extern void TF_SetAttrBool (TF_OperationDescription *desc, const char *attr_name, unsigned char value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrBool (TF_OperationDescription desc, string attr_name, byte value);

		public void SetAttr (string attrName, bool value)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrBool (handle, attrName, (byte)(value ? 1 : 0));
		}

		// extern void TF_SetAttrBoolList (TF_OperationDescription *desc, const char *attr_name, const unsigned char *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrBoolList (TF_OperationDescription desc, string attr_name, bool [] values, int num_values);

		public void SetAttr (string attrName, bool [] values)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrBoolList (handle, attrName, values, values.Length);
		}

		// extern void TF_SetAttrType (TF_OperationDescription *desc, const char *attr_name, TF_DataType value);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrType (TF_OperationDescription desc, string attr_name, TFDataType value);

		public void SetAttrType (string attrName, TFDataType dataType)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrType (handle, attrName, dataType);
		}

		// extern void TF_SetAttrTypeList (TF_OperationDescription *desc, const char *attr_name, const TF_DataType *values, int num_values);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTypeList (TF_OperationDescription desc, string attr_name, TFDataType [] values, int num_values);

		public void SetAttrType (string attrName, params TFDataType [] dataType)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (dataType == null)
				throw new ArgumentNullException (nameof (dataType));
			TF_SetAttrTypeList (handle, attrName, dataType, dataType.Length);
		}

		// extern void TF_SetAttrShape (TF_OperationDescription *desc, const char *attr_name, const int64_t *dims, int num_dims);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, long [] dims, int num_dims);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShape (TF_OperationDescription desc, string attr_name, IntPtr dims, int num_dims);

		public void SetAttrShape (string attrName, long [] dims)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (dims == null)
				TF_SetAttrShape (handle, attrName, null, -1);
			else
				TF_SetAttrShape (handle, attrName, dims, dims.Length);
		}

		// extern void TF_SetAttrShapeList (TF_OperationDescription *desc, const char *attr_name, const int64_t *const *dims, const int *num_dims, int num_shapes);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrShapeList (TF_OperationDescription desc, string attr_name, IntPtr dims, int [] num_dims, int num_shapes);

		public void SetAttrShape (string attrName, long [] [] dims)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (dims == null)
				throw new ArgumentNullException (nameof (dims));
			int num_shapes = dims.Length;
			var num_dims = new int [dims.Length];
			unsafe
			{
				var unmanaged = Marshal.AllocHGlobal (sizeof (IntPtr) * num_shapes);
				int ofs = 0;
				for (int i = 0; i < num_shapes; i++) {
					IntPtr array = Marshal.AllocHGlobal (sizeof (long) * dims [i].Length);
					Marshal.Copy (dims [i], 0, array, dims [i].Length);
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
		}

		// extern void TF_SetAttrTensorShapeProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorShapeProto (TF_OperationDescription desc, string attr_name, IntPtr proto, size_t proto_len, TF_Status status);
		public void SetAttrTensorShapeProto (string attrName, IntPtr proto, size_t protoLen, TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			TF_SetAttrTensorShapeProto (handle, attrName, proto, protoLen, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
		}

		// extern void TF_SetAttrTensorShapeProtoList (TF_OperationDescription *desc, const char *attr_name, const void *const *protos, const size_t *proto_lens, int num_shapes, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorShapeProtoList (TF_OperationDescription desc, string attr_name, void** protos, size_t* proto_lens, int num_shapes, TF_Status status);
		// TODO:

		// extern void TF_SetAttrTensor (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *value, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensor (TF_OperationDescription desc, string attr_name, TF_Tensor value, TF_Status status);

		public void SetAttr (string attrName, TFTensor tensor, TFStatus status = null)
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
		}

		// extern void TF_SetAttrTensorList (TF_OperationDescription *desc, const char *attr_name, TF_Tensor *const *values, int num_values, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrTensorList (TF_OperationDescription desc, string attr_name, IntPtr [] values, int num_values, TF_Status status);
		public void SetAttr (string attrName, TFTensor [] tensor, TFStatus status = null)
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
		}

		// extern void TF_SetAttrValueProto (TF_OperationDescription *desc, const char *attr_name, const void *proto, size_t proto_len, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_SetAttrValueProto (TF_OperationDescription desc, string attr_name, void* proto, size_t proto_len, TF_Status status);
		// TODO:

		// extern TF_Operation * TF_FinishOperation (TF_OperationDescription *desc, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Operation TF_FinishOperation (TF_OperationDescription desc, TF_Status status);

		public TFOperation FinishOperation (TFStatus status = null)
		{
			if (handle == IntPtr.Zero)
				ObjectDisposedException ();
			var cstatus = TFStatus.Setup (status);
			var h = TF_FinishOperation (handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			handle = IntPtr.Zero;
			GC.SuppressFinalize (this);

			return new TFOperation (graph, h);
		}
	}

	public partial class TFOperation
	{
		internal IntPtr handle;
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

		public TFOutput this [int idx] {
			get {
				return new TFOutput (this, idx);
			}
		}
	}

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


	}

	public class TFSession : TFDisposable
	{
		// extern TF_Session * TF_NewSession (TF_Graph *graph, const TF_SessionOptions *opts, TF_Status *status);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Session TF_NewSession (TF_Graph graph, TF_SessionOptions opts, TF_Status status);

		TFSession (IntPtr handle) : base (handle) { }

		public TFSession (TFGraph graph, TFSessionOptions sessionOptions, TFStatus status = null) : base (IntPtr.Zero)
		{
			var cstatus = TFStatus.Setup (status);
			var h = TF_NewSession (graph.handle, sessionOptions.handle, cstatus.handle);
			cstatus.CheckMaybeRaise (status);
			handle = h;
		}

		public TFSession (TFGraph graph, TFStatus status = null) : base (IntPtr.Zero)
		{
			var cstatus = TFStatus.Setup (status);
			var empty = TFSessionOptions.TF_NewSessionOptions ();
			var h = TF_NewSession (graph.handle, empty, cstatus.handle);
			TFSessionOptions.TF_DeleteSessionOptions (empty);
			cstatus.CheckMaybeRaise (status);
			handle = h;
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

				if (cstatus.CheckMaybeRaise (status))
					return new TFSession (h);
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

		#if false
		public struct Input
		{
			public TFOutput InputTF;
			public TFTensor Value;
			public Input (TFOutput input, TFTensor value)
			{
				InputTF = input;
				Value = value;
			}
		}

		public TFTensor [] Run (IEnumerable<Input> x)
		{
		// This API call would look liek this:
			Run (new [] { new Input (default (TFOutput), null) , new Input (default (TFOutput), null)});
		}

		#endif

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

		public struct PartialRunToken
		{
			internal IntPtr token;
		}

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
		static extern unsafe TFBuffer TF_GetOpList (TF_Library lib_handle);
		// TODO:

		// extern void TF_DeleteLibraryHandle (TF_Library *lib_handle);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe void TF_DeleteLibraryHandle (TF_Library lib_handle);

		internal override void NativeDispose (IntPtr handle)
		{
			TF_DeleteLibraryHandle (handle);
		}
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
	/// Represents a specific output of an operation
	/// </summary>
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
		/// <param name="index">The index of the output within the operation.</param>
		/// <remarks>
		/// This constructor is a low-level constructor used when you create operations
		/// manually using <see cref="T:TensorFlow.TFOperationDesc"/>, you typically
		/// create the outputs and pass these to the AddInput method to register the
		/// outputs in the operation.
		/// </remarks>
		public TFOutput (TFOperation operation, int index)
		{
			if (operation == null)
				throw new ArgumentNullException ("operation");
			LLOperation = operation.handle;
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

		public override string ToString ()
		{
			return string.Format ($"[TFAttributeMetadata IsList={IsList != 0?true:false} ListSize={ListSize} Type={Type} TotalSize={TotalSize}]");
		}
	}

}
