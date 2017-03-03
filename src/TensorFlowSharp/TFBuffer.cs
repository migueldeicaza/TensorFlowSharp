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
using size_t = System.UIntPtr;

namespace TensorFlow
{

	using static NativeMethods;

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

		internal TFBuffer (IntPtr handle) : base (handle)
		{
			
		}

		public unsafe TFBuffer () : base ((IntPtr)TF_NewBuffer ())
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
		public unsafe TFBuffer (IntPtr buffer, long size, BufferReleaseFunc release) : base ((IntPtr)TF_NewBuffer ())
		{
			var buf = (LLBuffer*)Handle;

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

	    private static IntPtr FreeBufferFunc;

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
				var buf = LLBuffer;
				buf->data = Marshal.AllocHGlobal (count);
				Marshal.Copy (buffer, start, buf->data, count);
				buf->length = (size_t)count;
				buf->data_deallocator = FreeBufferFunc;
			}
		}

		internal unsafe LLBuffer* LLBuffer => (LLBuffer*)Handle;



		protected override void NativeDispose (IntPtr handle)
		{
			unsafe { TF_DeleteBuffer ((LLBuffer*)handle); }
		}

		/// <summary>
		/// Returns a byte array representing the data wrapped by this buffer.
		/// </summary>
		/// <returns>The array.</returns>
		public byte [] ToArray ()
		{
			if (Handle == IntPtr.Zero)
				return null;

			unsafe
			{
				var lb = (LLBuffer*)Handle;

				var result = new byte [(int)lb->length];
				Marshal.Copy (lb->data, result, 0, (int)lb->length);

				return result;
			}
		}
	}
}