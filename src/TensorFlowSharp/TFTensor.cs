//
// TensorFlow API for .NET languages
// https://github.com/migueldeicaza/TensorFlowSharp
//
// Authors:
//    Miguel de Icaza (miguel@microsoft.com)
//    Gustavo J Knuppe (https://github.com/knuppe/)
//

using System;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using size_t = System.UIntPtr;

#pragma warning disable XS0001 // Find usages of mono todo items

namespace TensorFlow
{

	using static NativeMethods;

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
	public sealed class TFTensor : TFDisposable
	{

		/// <summary>
		/// Signature that methods must conform to to be used to release memory that was passed to a manually allocated TFTensor
		/// </summary>
		public delegate void Deallocator (IntPtr data, IntPtr size, IntPtr deallocatorData);

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

		/// <summary>
		/// Creates a new tensor from a portion of an array of sbytes
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, sbyte [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Int8, shape, data, start, count, 2));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of bytes
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, byte [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.UInt8, shape, data, start, count, 1));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of shorts
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, short [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Int16, shape, data, start, count, 2));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of ushorts
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, ushort [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.UInt16, shape, data, start, count, 2));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of ints
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, int [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Int32, shape, data, start, count, 4));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of floats
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, float [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Float, shape, data, start, count, 4));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of doubles
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, double [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Double, shape, data, start, count, 8));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of longs
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, long [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Int64, shape, data, start, count, 8));
		}
		
		/// <summary>
		/// Creates a new tensor from a portion of an array of Complex numbers
		/// </summary>
		/// <param name="shape">Represents the tensor shape.</param>
		/// <param name="data">The linear array of data, the data is shuffled to fit in the tensor with the specified dimensions.</param>
		/// <param name="start">The offset into the provided data array where the data resides.</param>
		/// <param name="count">The number of bytes to copy from count into the tensor.</param>
		/// <remarks>
		/// Use the FromBuffer method to create a tensor that has the specified dimensions
		/// and is initialized with data from the data array.   The data is copied starting
		/// at the start offset, for count bytes and is laid out into the tensor following the
		/// specified dimensions.
		/// </remarks>
		public static TFTensor FromBuffer (TFShape shape, Complex [] data, int start, int count)
		{
			return new TFTensor (SetupTensor (TFDataType.Complex128, shape, data, start, count, 16));
		}
		

		/// <summary>
		/// Creates a constant tensor with a single dimension from an integer value.
		/// </summary>
		public unsafe TFTensor (int value)
		{
			var v = (int*)Marshal.AllocHGlobal (sizeof (int));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Int32, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (int), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an sbyte value.
		/// </summary>
		public unsafe TFTensor (sbyte value)
		{
			var v = (sbyte*)Marshal.AllocHGlobal (sizeof (sbyte));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Int8, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (sbyte), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a short value.
		/// </summary>
		public unsafe TFTensor (short value)
		{
			var v = (short*)Marshal.AllocHGlobal (sizeof (short));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Int16, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (short), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an ushort value.
		/// </summary>
		public unsafe TFTensor (ushort value)
		{
			var v = (ushort*)Marshal.AllocHGlobal (sizeof (ushort));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Int16, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (ushort), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an byte value.
		/// </summary>
		public unsafe TFTensor (byte value)
		{
			var v = (int*)Marshal.AllocHGlobal (sizeof (byte));
			*v = value;
			Handle = TF_NewTensor (TFDataType.UInt8, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (byte), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a Complex value.
		/// </summary>
		public unsafe TFTensor (Complex value)
		{
			var v = (Complex*)Marshal.AllocHGlobal (sizeof (Complex));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Complex128, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (Complex), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a float value.
		/// </summary>
		public unsafe TFTensor (float value)
		{
			var v = (float*)Marshal.AllocHGlobal (sizeof (float));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Float, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (float), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a double value.
		/// </summary>
		public unsafe TFTensor (double value)
		{
			var v = (double*)Marshal.AllocHGlobal (sizeof (double));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Double, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (double), FreeTensorData, IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a long value.
		/// </summary>
		public unsafe TFTensor (long value)
		{
			var v = (long*)Marshal.AllocHGlobal (sizeof (long));
			*v = value;
			Handle = TF_NewTensor (TFDataType.Int64, IntPtr.Zero, 0, (IntPtr)v, (size_t)sizeof (long), FreeTensorData, IntPtr.Zero);
		}

		// Convenience, should I add T[,] and T[,,] as more convenience ones?
		public TFTensor (sbyte [] data) : base (SetupTensor (TFDataType.Int8, data, 2)) { }
		public TFTensor (byte [] data) : base (SetupTensor (TFDataType.UInt8, data, 1)) { }
		public TFTensor (short [] data) : base (SetupTensor (TFDataType.Int16, data, 2)) { }
		public TFTensor (ushort [] data) : base (SetupTensor (TFDataType.UInt16, data, 2)) { }
		public TFTensor (int [] data) : base (SetupTensor (TFDataType.Int32, data, 4)) { }
		public TFTensor (float [] data) : base (SetupTensor (TFDataType.Float, data, 4)) { }
		public TFTensor (double [] data) : base (SetupTensor (TFDataType.Double, data, 8)) { }
		public TFTensor (long [] data) : base (SetupTensor (TFDataType.Int64, data, 8)) { }
		public TFTensor (Complex [] data) : base (SetupTensor (TFDataType.Complex128, data, 16)) { }

		/// <summary>
		/// Creates a single-dimension tensor from a byte buffer.  This is different than creating a tensor from a byte array that produces a tensor with as many elements as the byte array.
		/// </summary>
		public static unsafe TFTensor CreateString (byte [] buffer)
		{
			if (buffer == null)
				throw new ArgumentNullException (nameof (buffer));
			//
			// TF_STRING tensors are encoded with a table of 8-byte offsets followed by
			// TF_StringEncode-encoded bytes.
			//
			var size = TF_StringEncodedSize ((size_t)buffer.Length);
			var handle = TF_AllocateTensor (TFDataType.String, IntPtr.Zero, 0, (size_t)((ulong)size + 8));

			// Clear offset table
			var dst = TF_TensorData (handle);
			Marshal.WriteInt64 (dst, 0);
			var status = TF_NewStatus ();
			fixed (byte* src = &buffer [0])
			{
				TF_StringEncode (src, (size_t)buffer.Length, (sbyte*)(dst + 8), size, status);
				var ok = TF_GetCode (status) == TFCode.Ok;
				TF_DeleteStatus (status);
				if (!ok)
					return null;
			}
			return new TFTensor (handle);
		}

		// Convenience function to factor out the setup of a new tensor from an array
	    private static IntPtr SetupTensor (TFDataType dt, long [] dims, Array data, int size)
		{
			return SetupTensor (dt, dims, data, 0, data.Length, size);
		}

		// Convenience function to factor out the setup of a new tensor from an array
	    private static IntPtr SetupTensor (TFDataType dt, Array data, int size)
		{
			var dims = new long [data.Rank];
			for (var i = 0; i < dims.Length; i++)
				dims [i] = data.GetLength (i);

			return SetupTensor (dt, dims, data, 0, data.Length, size);
		}

		// Use for single dimension arrays 
	    private static IntPtr SetupTensor (TFDataType dt, TFShape shape, Array data, int start, int count, int size)
		{
			if (shape == null)
				throw new ArgumentNullException (nameof (shape));
			return SetupTensor (dt, shape.dims, data, start, count, size);
		}
		
		// Use for single dimension arrays 
	    private static IntPtr SetupTensor (TFDataType dt, long [] dims, Array data, int start, int count, int size)
		{
			if (start < 0 || start > data.Length - count)
				throw new ArgumentException ("start + count > Array size");

			var dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned);

			if (dims == null)
				return TF_NewTensor (dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject () + start * size, (size_t)(count * size), FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
			
			return TF_NewTensor (dt, dims, dims.Length, dataHandle.AddrOfPinnedObject () + start * size, (size_t)(count * size), FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
		}

		// Use for multiple dimension arrays 
	    private static IntPtr SetupMulti (TFDataType dt, long [] dims, Array data, long bytes)
		{
			var dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned);

			if (dims == null)
				return TF_NewTensor (dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject (), (size_t)bytes, FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
			
			return TF_NewTensor (dt, dims, dims.Length, dataHandle.AddrOfPinnedObject (), (size_t)bytes, FreeTensorHandle, GCHandle.ToIntPtr (dataHandle));
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

	    public static implicit operator TFTensor (double value)
		{
			return new TFTensor (value);
		}

	    public static implicit operator TFTensor (float value)
		{
			return new TFTensor (value);
		}

	    public static implicit operator TFTensor (Complex value)
		{
			return new TFTensor (value);
		}

	    public static implicit operator TFTensor (byte value)
		{
			return new TFTensor (value);
		}

	    public static implicit operator TFTensor (Array array)
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
				if (t == typeof(Complex)) {
					size = 16;
					dt = TFDataType.Complex128;
				} else
					throw new ArgumentException ($"The data type {t} is not supported");
				break;
			}

			var dims = new long [array.Rank];
			for (var i = 0; i < array.Rank; i++) {
				dims [i] = array.GetLength (i);
				size *= (int)dims [i];
			}
			var newTensor = new TFTensor (SetupMulti (dt, dims, array, size));
			return newTensor;
		}

		// General purpose constructor, specifies data type and gets pointer to buffer
		// Is the default good, one where we let the user provide their own deallocator, or should we make a copy in that case?
		public TFTensor (TFDataType dataType, long [] dims, IntPtr data, size_t dataSize, Deallocator deallocator, IntPtr deallocatorData) 
		{
			if (dims == null)
				throw new ArgumentNullException (nameof(dims));

			Handle = TF_NewTensor (dataType, dims, dims.Length, data, dataSize, deallocator, deallocatorData);

		}

		protected override void NativeDispose (IntPtr handle)
		{
			TF_DeleteTensor (handle);
		}

		public TFTensor (TFDataType dataType, long [] dims, int size) 
		{
			if (dims == null)
				throw new ArgumentNullException (nameof(dims));
			
			Handle = TF_AllocateTensor (dataType, dims, dims.Length, (size_t)size);
		}

		public TFDataType TensorType => TF_TensorType (Handle);


		/// <summary>
		/// Returns the number of dimensions in the tensor.
		/// </summary>
		/// <remarks>
		/// For single-dimension tensors the return is 1, 2 dimensions is 2 and so on.
		/// </remarks>
		public int NumDims => TF_NumDims (Handle);

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
			return TF_Dim (Handle, dimIndex);
		}


		public size_t TensorByteSize => TF_TensorByteSize (Handle);

		/// <summary>
		/// Returns a pointer to the raw data in the tensor.
		/// </summary>
		/// <remarks>
		/// The contents of the Data must be interpreted according to the type of the
		/// data as described by the DataType property.   The amount of data
		/// is given by the the TensorByteSize property.
		/// </remarks>
		public IntPtr Data => TF_TensorData (Handle);

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
				var dims = new long [TF_NumDims (Handle)];
				for (var i = 0; i < dims.Length; i++)
					dims [i] = (int)TF_Dim (Handle, i);

				return dims;
			}
		}

	    private static Type TypeFromTensorType (TFDataType type)
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

	    private static unsafe object FetchSimple (TFDataType dt, IntPtr data)
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

	    private static unsafe void Copy (IntPtr src, void* target, int size)
		{
			Buffer.MemoryCopy ((void*)src, target, size, size);
		}

	    private static unsafe void FetchFlatArray (Array target, TFDataType dt, IntPtr data)
		{
			var len = target.Length;
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
					Copy (data, p, len * sizeof (float));
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

	    private static unsafe object FetchJaggedArray (Type t, TFDataType dt, ref IntPtr data, long [] shape, int level = 0)
		{
			Array target;

			// If we are at the last node
			if (level == shape.Length - 1) {
				target = Array.CreateInstance (t, shape [level]);

				for (long l = 0; l < shape [level]; l++)
					switch (dt) {
					case TFDataType.Float:
						target.SetValue (*(float*)data, l);
						data += 4;
						break;
					case TFDataType.Double:
						target.SetValue (*(double*)data, l);
						data += 8;
						break;
					case TFDataType.Int32:
						target.SetValue (*(int*)data, l);
						data += 4;
						break;
					case TFDataType.UInt8:
						target.SetValue (*(byte*)data, l);
						data += 1;
						break;
					case TFDataType.Int16:
						target.SetValue (*(short*)data, l);
						data += 2;
						break;
					case TFDataType.Int8:
						target.SetValue (*(sbyte*)data, l);
						data += 1;
						break;
					case TFDataType.Int64:
						target.SetValue (*(long*)data, l);
						data += 8;
						break;
					case TFDataType.Bool:
						target.SetValue (*(bool*)data, l);
						data += 1;
						break;
					case TFDataType.Complex128:
						target.SetValue (*(Complex*)data, l);
						data += sizeof (Complex);
						break;
					case TFDataType.String:
						throw new NotImplementedException ("String decoding not implemented for tensor vecotrs yet");
					default:
						throw new NotImplementedException ();
					}
			} else {
				target = null;

				var top = shape [level];
				if (top < int.MaxValue) {
					var itop = (int)top;

					for (var i = 0; i < itop; i++) {
						var childArray = FetchJaggedArray (t, dt, ref data, shape, level + 1);
						if (target == null)
							target = Array.CreateInstance (childArray.GetType (), shape [level]);

						target.SetValue (childArray, i);
					}
				} else {
					for (var l = 0; l < top; l++) {

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

	    private static void FetchMultiDimensionalArray (Array target, TFDataType dt, IntPtr data, long [] shape)
		{
			var idx = new int [shape.Length];
			for (var i = 0; i < shape.Length; i++) {
				if (shape [i] > int.MaxValue)
					throw new ArgumentOutOfRangeException (nameof(shape), "Shape can not be longer than 32 bits");
				
				idx [i] = (int)shape [i];
			}
			Copy (target, dt, shape, idx, shape.Length - 1, ref data);
		}

	    private static unsafe void Copy (Array target, TFDataType dt, long [] shape, int [] idx, int level, ref IntPtr data)
		{
			if (level > 0) {
				for (idx [level] = 0; idx [level] < shape [level]; idx [level]++)
					Copy (target, dt, shape, idx, level - 1, ref data);
			} else {
				for (idx [0] = 0; idx [0] < shape [0]; idx [0]++) {
					switch (dt) {
					case TFDataType.Float:
						target.SetValue (*(float*)data, idx);
						data += 4;
						break;
					case TFDataType.Double:
						target.SetValue (*(double*)data, idx);
						data += 8;
						break;
					case TFDataType.Int32:
						target.SetValue (*(int*)data, idx);
						data += 4;
						break;
					case TFDataType.UInt8:
						target.SetValue (*(byte*)data, idx);
						data += 1;
						break;
					case TFDataType.Int16:
						target.SetValue (*(short*)data, idx);
						data += 2;
						break;
					case TFDataType.Int8:
						target.SetValue (*(sbyte*)data, idx);
						data += 1;
						break;
					case TFDataType.Int64:
						target.SetValue (*(long*)data, idx);
						data += 8;
						break;
					case TFDataType.Bool:
						target.SetValue (*(bool*)data, idx);
						data += 1;
						break;
					case TFDataType.Complex128:
						target.SetValue (*(Complex*)data, idx);
						data += sizeof (Complex);
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

			if (dims == 1) {
				var r1 = Array.CreateInstance (t, Shape [0]);
				FetchFlatArray (r1, TensorType, Data);
				return r1;
			}

			if (jagged) {
				var data = Data;
				return FetchJaggedArray (t, TensorType, ref data, Shape);
			}
		    var result = Array.CreateInstance (t, Shape);
		    FetchMultiDimensionalArray (result, TensorType, Data, Shape);
		    return result;
		}

		public override string ToString ()
		{
			var n = NumDims;
			if (n == 0)
				return GetValue ().ToString ();

			var sb = new StringBuilder ("[");
			for (var i = 0; i < n; i++) {
				sb.Append (TF_Dim (Handle, i));
				if (i + 1 < n)
					sb.Append ("x");
			}
			sb.Append ("]");
			return sb.ToString ();
		}
	}
}	