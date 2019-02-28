//
// TensorFlow.cs; Bindings to the TensorFlow C API for .NET
// 
// Authors:
//   Miguel de Icaza (miguel@microsoft.com)
//
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using size_t = System.UIntPtr;
using TF_Tensor = System.IntPtr;

namespace TensorFlow
{

	/// <summary>
	/// TFTensor holds a multi-dimensional array of elements of a single data type.
	/// </summary>
	/// <remarks>
	/// <para>
	/// You can create tensors with the various constructors in this class, or using
	/// the implicit conversions from various data types into a TFTensor, including
	/// the creation of tensors from simple constants (returning a tensor that reprensets
	/// a scalar, that is, it is a 0D tensor), arrays (returning a tensor of a single
	/// dimension, 1D) or arbitrary multidimensional arrays.
	///</para>
	/// <para>
	///   Given a tensor, you can retrieve the number of dimensions in it via the
	///   NumDims property, or you can retrieve the shape of a tensor, that is how many
	///   elements on each dimension the tensor has, by fetching the Shape property.
	/// </para>
	/// <para>
	/// The implicit conversions for basic types produce tensors of one dimesion with
	/// a single element, while the implicit conversion from an array, expects a multi-dimensional
	/// array that is converted into a tensor of the right dimensions.
	/// </para>
	/// <para>
	/// The special "String" tensor data type that you will find in TensorFlow documentation
	/// really represents a byte array.   You can create string tensors by using the <see cref="M:TensorFlow.TFTensor.CreateString"/> 
	/// method that takes a byte array buffer as input.
	/// </para>
	/// <example>
	/// <code>
	///   TFTensor scalar = 1;           // Creates a 0D tensor, for the integer value 1
	///   int d = scalar.NumDims;        // d will be equal to zero, as it is a 0D tensor
	///   long [] shape = scalar.Shape   // returns an empty array, as it is a 0D tensor
	///   
	///   TFTensor list = new [] {1,2,3} // Creates a 1D tensor, or vector, for the values 1, 2, 3
	///   d = list.NumDims;              // d will be one
	///   shape = list.Shape;            // shape will be an array with a single value 3, representing that the dimension 0 has 3 elements
	/// 
	///                                  // Creates a 3D tensor, 
	///   TFTensor cube = new [,,] { {{1,2,3},{4,5,6}}}
	///   d = cube.NumDims               // d will be 3
	///   shape = list.Shape             // shape will be [1,2,3] which is the shape of the above 3D array
	/// </code>
	/// </example>
	/// </remarks>
	public class TFTensor : TFDisposableThreadSafe
	{
		/// <summary>
		/// Signature that methods must conform to to be used to release memory that was passed to a manually allocated TFTensor
		/// </summary>
		public delegate void Deallocator (IntPtr data, IntPtr size, IntPtr deallocatorData);

		// extern TF_Tensor * TF_NewTensor (TF_DataType, const int64_t *dims, int num_dims, void *data, size_t len, void (* deallocator)(void *, size_t, void *), void *deallocator_arg);
		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_NewTensor (TFDataType dataType, long [] dims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg);

		[DllImport (NativeBinding.TensorFlowLibrary)]
		static extern unsafe TF_Tensor TF_NewTensor (TFDataType dataType, IntPtr zeroDims, int num_dims, IntPtr data, size_t len, Deallocator deallocator, IntPtr deallocator_arg);

		internal TFTensor (IntPtr handle) : base (handle) { }

		static Deallocator FreeTensorDataDelegate = FreeTensorData;
		static Deallocator FreeTensorHandleDelegate = FreeTensorHandle;
		
		[MonoPInvokeCallback (typeof (Deallocator))]
		internal static void FreeTensorData (IntPtr data, IntPtr len, IntPtr closure)
		{
			Marshal.FreeHGlobal (data);
		}

		[MonoPInvokeCallback (typeof (Deallocator))]
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
			return new TFTensor (SetupTensor (TFDataType.Int8, shape, data, start, count, size: 2));
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
			return new TFTensor (SetupTensor (TFDataType.UInt8, shape, data, start, count, size: 1));
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
			return new TFTensor (SetupTensor (TFDataType.Int16, shape, data, start, count, size: 2));
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
			return new TFTensor (SetupTensor (TFDataType.UInt16, shape, data, start, count, size: 2));
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
			return new TFTensor (SetupTensor (TFDataType.Int32, shape, data, start, count, size: 4));
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
			return new TFTensor (SetupTensor (TFDataType.Float, shape, data, start, count, size: 4));
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
			return new TFTensor (SetupTensor (TFDataType.Double, shape, data, start, count, size: 8));
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
			return new TFTensor (SetupTensor (TFDataType.Int64, shape, data, start, count, size: 8));
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
			return new TFTensor (SetupTensor (TFDataType.Complex128, shape, data, start, count, size: 16));
		}

		/// <summary>
		/// Creates a constant tensor from an array, the shape reflects the shape of the C# array and the underlying type reflects the C# type.
		/// </summary>
		public unsafe TFTensor (Array array)
		{
			if (array == null)
				throw new ArgumentNullException (nameof (array));

			// Ensure that, if we have arrays of arrays, we can handle them accordingly:
			if (isJagged (array.GetType ())) {
				Type elementType = getInnerMostType (array);
				int [] length = getLength (array);
				Array multidimensional = Array.CreateInstance (elementType, length);
				Array flatten = deepFlatten (array);
				Buffer.BlockCopy (flatten, 0, multidimensional, 0, flatten.Length * Marshal.SizeOf (elementType));
				createFromMultidimensionalArrays (multidimensional);
			} else {
				createFromMultidimensionalArrays (array);
			}
		}

		private unsafe void createFromMultidimensionalArrays (Array array)
		{
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
				if (t.IsAssignableFrom (typeof (Complex))) {
					size = 16;
					dt = TFDataType.Complex128;
				} else
					throw new ArgumentException ($"The data type {t} is not supported");
				break;
			}

			var dims = new long [array.Rank];
			for (int i = 0; i < array.Rank; i++) {
				dims [i] = array.GetLength (i);
				size *= (int)dims [i];
			}
			handle = SetupMulti (dt, dims, array, size);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an integer value.
		/// </summary>
		public unsafe TFTensor (int value)
		{
			var v = (int*)Marshal.AllocHGlobal (sizeof (int));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int32, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (int), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a boolean value.
		/// </summary>
		public unsafe TFTensor (bool value)
		{
			var v = (bool*)Marshal.AllocHGlobal (sizeof (bool));
			*v = value;
			handle = TF_NewTensor (TFDataType.Bool, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (int), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an sbyte value.
		/// </summary>
		public unsafe TFTensor (sbyte value)
		{
			var v = (sbyte*)Marshal.AllocHGlobal (sizeof (sbyte));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int8, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (sbyte), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a short value.
		/// </summary>
		public unsafe TFTensor (short value)
		{
			var v = (short*)Marshal.AllocHGlobal (sizeof (short));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int16, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (short), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an ushort value.
		/// </summary>
		public unsafe TFTensor (ushort value)
		{
			var v = (ushort*)Marshal.AllocHGlobal (sizeof (ushort));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int16, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (ushort), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from an byte value.
		/// </summary>
		public unsafe TFTensor (byte value)
		{
			var v = (int*)Marshal.AllocHGlobal (sizeof (byte));
			*v = value;
			handle = TF_NewTensor (TFDataType.UInt8, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (byte), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a Complex value.
		/// </summary>
		public unsafe TFTensor (Complex value)
		{
			var v = (Complex*)Marshal.AllocHGlobal (sizeof (Complex));
			*v = value;
			handle = TF_NewTensor (TFDataType.Complex128, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (Complex), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a float value.
		/// </summary>
		public unsafe TFTensor (float value)
		{
			var v = (float*)Marshal.AllocHGlobal (sizeof (float));
			*v = value;
			handle = TF_NewTensor (TFDataType.Float, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (float), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a double value.
		/// </summary>
		public unsafe TFTensor (double value)
		{
			var v = (double*)Marshal.AllocHGlobal (sizeof (double));
			*v = value;
			handle = TF_NewTensor (TFDataType.Double, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (double), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		/// <summary>
		/// Creates a constant tensor with a single dimension from a long value.
		/// </summary>
		public unsafe TFTensor (long value)
		{
			var v = (long*)Marshal.AllocHGlobal (sizeof (long));
			*v = value;
			handle = TF_NewTensor (TFDataType.Int64, zeroDims: IntPtr.Zero, num_dims: 0, data: (IntPtr)v, len: (UIntPtr)sizeof (long), deallocator: FreeTensorDataDelegate, deallocator_arg: IntPtr.Zero);
		}

		// Convenience, should I add T[,] and T[,,] as more convenience ones?

		/// <summary>
		/// Creates a 1 dimensional tensor from an array of booleans.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (bool [] data) : base (SetupTensor (TFDataType.Bool, data, size: 1)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of sbytes.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (sbyte [] data) : base (SetupTensor (TFDataType.Int8, data, size: 1)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of bytes.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (byte [] data) : base (SetupTensor (TFDataType.UInt8, data, size: 1)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of shorts.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (short [] data) : base (SetupTensor (TFDataType.Int16, data, size: 2)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of ushorts
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (ushort [] data) : base (SetupTensor (TFDataType.UInt16, data, size: 2)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of ints.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (int [] data) : base (SetupTensor (TFDataType.Int32, data, size: 4)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of floats.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (float [] data) : base (SetupTensor (TFDataType.Float, data, size: 4)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of doubles.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (double [] data) : base (SetupTensor (TFDataType.Double, data, size: 8)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of longs.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (long [] data) : base (SetupTensor (TFDataType.Int64, data, size: 8)) { }
		/// <summary>
		/// Creates a 1 dimensional tensor from an array of complex numbers.
		/// </summary>
		/// <param name="data">Data.</param>
		public TFTensor (Complex [] data) : base (SetupTensor (TFDataType.Complex128, data, size: 16)) { }

        /// <summary>
        /// Creates a single-dimension tensor from a byte buffer.  This is different than creating a tensor from a byte array that produces a tensor with as many elements as the byte array.
        /// </summary>
        public unsafe static TFTensor CreateString(byte[] buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by
            // TF_StringEncode-encoded bytes.
            //
            var size = TFString.TF_StringEncodedSize((UIntPtr)buffer.Length);
            IntPtr handle = TF_AllocateTensor(TFDataType.String, IntPtr.Zero, 0, (UIntPtr)((ulong)size + 8));

            // Clear offset table
            IntPtr dst = TF_TensorData(handle);
            Marshal.WriteInt64(dst, 0);
            var status = TFStatus.TF_NewStatus();
            fixed (byte* src = &buffer[0])
            {
                TFString.TF_StringEncode(src, (UIntPtr)buffer.Length, (byte*)(dst + 8), size, status);
                var ok = TFStatus.TF_GetCode(status) == TFCode.Ok;
                TFStatus.TF_DeleteStatus(status);
                if (!ok)
                    return null;
            }
            return new TFTensor(handle);
        }

        /// <summary>
		/// Converts a single-dimension tensor into a byte buffer. The byte array can be further decoded into strings using appropriate encoding scheme e.g. "UTF8"
		/// </summary>
        public static unsafe byte[] DecodeString(TFTensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            var src = TF_TensorData(tensor.handle);
            using (var status = new TFStatus())
            {
                IntPtr dst = IntPtr.Zero;
                UIntPtr dst_len = UIntPtr.Zero;
                TFString.TF_StringDecode((byte*)(src + 8), tensor.TensorByteSize - 8, (byte**)&dst, &dst_len, status.handle);
                var ok = status.StatusCode == TFCode.Ok;
                if (!ok)
                    return null;
                var buffer = new byte[(int)dst_len];
                Marshal.Copy(dst, buffer, 0, buffer.Length);
                return buffer;
            }
        }

        /// <summary>
		/// Creates a multi-dimension tensor from an array of byte buffer. The bytes for string[i] are represented as buffer[i][:].
		/// </summary>
        public static unsafe TFTensor CreateString(byte[][] buffer, TFShape shape)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            int size = 0;
            foreach (var b in buffer)
            {
                size += (int)TFString.TF_StringEncodedSize((UIntPtr)b.Length);
            }
            int totalSize = size + buffer.Length * 8;
            ulong offset = 0;
            IntPtr handle = TF_AllocateTensor(TFDataType.String, shape.dims, shape.dims.Length, (UIntPtr)totalSize);

            // Clear offset table
            IntPtr pOffset = TF_TensorData(handle);
            IntPtr dst = pOffset + buffer.Length * 8;
            IntPtr dstLimit = pOffset + totalSize;
            for (int i = 0; i < buffer.Length; i++)
            {
                Marshal.WriteInt64(pOffset, (long)offset);
                using (var status = new TFStatus())
                {
                    fixed (byte* src = &buffer[i][0])
                    {
                        var written = TFString.TF_StringEncode(src, (UIntPtr)buffer[i].Length, (byte*)dst, (size_t)(dstLimit.ToInt64() - dst.ToInt64()), status.handle);
                        var ok = status.StatusCode == TFCode.Ok;
                        if (!ok)
                            return null;
                        pOffset += 8;
                        dst += (int)written;
                        offset += written.ToUInt64();
                    }
                }
            }
            return new TFTensor(handle);
        }

        /// <summary>
		/// Converts a multi-dimension tensor into a byte buffer array. The byte array can be further decoded into strings using appropriate encoding scheme e.g. "UTF8"
		/// </summary>
        public static unsafe byte[][] DecodeMultiDimensionString(TFTensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            //
            // TF_STRING tensors are encoded with a table of 8-byte offsets followed by TF_StringEncode-encoded bytes.
            // [offset1, offset2,...,offsetn, s1size, s1bytes, s2size, s2bytes,...,snsize,snbytes]
            //
            long size = 1;
            foreach (var s in tensor.Shape)
                size *= s;

            var buffer = new byte[size][];
            var src = TF_TensorData(tensor.handle);
            var srcLen = (IntPtr)(src.ToInt64() + (long)tensor.TensorByteSize);
            src += (int)(size * 8);
            for (int i = 0; i < buffer.Length; i++)
            {
                using (var status = new TFStatus())
                {
                    IntPtr dst = IntPtr.Zero;
                    UIntPtr dstLen = UIntPtr.Zero;
                    var read = TFString.TF_StringDecode((byte*)src, (size_t)(srcLen.ToInt64() - src.ToInt64()), (byte**)&dst, &dstLen, status.handle);
                    var ok = status.StatusCode == TFCode.Ok;
                    if (!ok)
                        return null;
                    buffer[i] = new byte[(int)dstLen];
                    Marshal.Copy(dst, buffer[i], 0, buffer[i].Length);
                    src += (int)read;
                }
            }
            return buffer;
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
		static IntPtr SetupTensor (TFDataType dt, TFShape shape, Array data, int start, int count, int size)
		{
			if (shape == null)
				throw new ArgumentNullException (nameof (shape));
			return SetupTensor (dt, shape.dims, data, start, count, size);
		}
		
		// Use for single dimension arrays 
		static IntPtr SetupTensor (TFDataType dt, long [] dims, Array data, int start, int count, int size)
		{
			if (start < 0 || start > data.Length - count)
				throw new ArgumentException ("start + count > Array size");

			var dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned);

			if (dims == null)
				return TF_NewTensor (dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject () + start * size, (UIntPtr)(count * size), FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle));
			else
				return TF_NewTensor (dt, dims, dims.Length, dataHandle.AddrOfPinnedObject () + start * size, (UIntPtr)(count * size), FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle));
		}

		// Use for multiple dimension arrays 
		static IntPtr SetupMulti (TFDataType dt, long [] dims, Array data, long bytes)
		{
			var dataHandle = GCHandle.Alloc (data, GCHandleType.Pinned);

			if (dims == null)
				return TF_NewTensor (dt, IntPtr.Zero, 0, dataHandle.AddrOfPinnedObject (), (UIntPtr)bytes, FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle));
			else
				return TF_NewTensor (dt, dims, dims.Length, dataHandle.AddrOfPinnedObject (), (UIntPtr)bytes, FreeTensorHandleDelegate, GCHandle.ToIntPtr (dataHandle));
		}

		// 
		// Factory methods to create tensors from a constant
		//

		/// <summary>
		/// Converts an integer into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the integer value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		public static implicit operator TFTensor (int value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a boolean into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the integer value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		public static implicit operator TFTensor (bool value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a long into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the long value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		public static implicit operator TFTensor (long value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a double into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the double value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		unsafe public static implicit operator TFTensor (double value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a float into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the float value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		unsafe public static implicit operator TFTensor (float value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a Complex number into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the complex value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		unsafe public static implicit operator TFTensor (Complex value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a byte into a 1-dimensional, 1-valued tensor.
		/// </summary>
		/// <returns>The tensor representing the byte value.</returns>
		/// <param name="value">Value to initialize the tensor with.</param>
		unsafe public static implicit operator TFTensor (byte value)
		{
			return new TFTensor (value);
		}

		/// <summary>
		/// Converts a C# array into a tensor.
		/// </summary>
		/// <returns>The tensor containing the data.</returns>
		/// <param name="array">single dimension, or multi-dimensional array.</param>
		/// <remarks>
		/// This implicit conversion can convert single or multidimensional arrays of
		/// booleans, sbytes, byte, shorts, ushorts, ints, longs, doubles, floats and
		/// complex numbers into a tensor with the same dimensional shape as the provided
		/// array.
		/// </remarks>
		unsafe public static implicit operator TFTensor (Array array)
		{
			return new TFTensor (array);

		}

		// General purpose constructor, specifies data type and gets pointer to buffer
		// Is the default good, one where we let the user provide their own deallocator, or should we make a copy in that case?
		/// <summary>
		/// Low-level tensor constructor that creates a tensor from a buffer pointed to by an IntPtr.
		/// </summary>
		/// <param name="dataType">Specifies the data type held by the tensor, as well as how to interpret the provided data.</param>
		/// <param name="dims">Describes the tensor shape, an array that indicates .</param>
		/// <param name="data">Pointer to the raw data that will be used to initialize the tensor.</param>
		/// <param name="dataSize">The size of the data being passed in.</param>
		/// <param name="deallocator">Deallocator method, it is invoked when the tensor is destroyed to release the data pointed to by <paramref name="data"/>.   On platforms like iOS (or other static compilation platforms), yiou must annotate the method specified in the deallocator with a <see cref="T:TensorFlow.MonoPInvokeCallbackAttribute"/>.</param>
		/// <param name="deallocatorData">An optional argument of data that is passed to the deallocator method when the tensor is destroyed, you can use this to pass context information.</param>
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

		/// <summary>
		/// Low-level: Creates an empty tensor of the specified type and shape, with the specified number of elements
		/// </summary>
		/// <param name="dataType">Data type.</param>
		/// <param name="dims">Tensor shape.</param>
		/// <param name="size">Size in bytes of the tensor, this will be the actual memory allocated.</param>
		/// <remarks>
		/// It is the responsibility of the caller to ensure that the size is correct given the data type size
		/// and the tensor dimension specified in dims.
		/// </remarks>
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

		/// <summary>
		/// Returns the data type for the tensor.
		/// </summary>
		/// <value>The type of the tensor.</value>
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
					dims [i] = (int)TF_Dim (handle, i);

				return dims;
			}
		}

		/// <summary>
		/// Converts a <see cref="TFDataType"/> to a system type.
		/// </summary>
		/// <param name="type">The <see cref="TFDataType"/> to be converted.</param>
		/// <returns>The system type corresponding to the given <paramref name="type"/>.</returns>
		public static Type TypeFromTensorType (TFDataType type)
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

		/// <summary>
		/// Converts a system type to a <see cref="TFDataType"/>.
		/// </summary>
		/// <param name="type">The system type to be converted.</param>
		/// <returns>The <see cref="TFDataType"/> corresponding to the given type.</returns>
		public static TFDataType TensorTypeFromType (Type type)
		{
			if (type == typeof (float))
				return TFDataType.Float;
			if (type == typeof (double))
				return TFDataType.Double;
			if (type == typeof (int))
				return TFDataType.Int32;
			if (type == typeof (byte))
				return TFDataType.UInt8;
			if (type == typeof (short))
				return TFDataType.Int16;
			if (type == typeof (sbyte))
				return TFDataType.Int8;
			if (type == typeof (string))
				return TFDataType.String;
			if (type == typeof (long))
				return TFDataType.Int64;
			if (type == typeof (bool))
				return TFDataType.Bool;
			if (type == typeof (ushort))
				return TFDataType.UInt16;
			if (type == typeof (Complex))
				return TFDataType.Complex128;

			throw new ArgumentOutOfRangeException (nameof(type), $"The given type could not be mapped to an existing {nameof(TFDataType)}.");
		}

		internal static (TFDataType dt, long size) TensorTypeAndSizeFromType (Type t)
		{
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
				if (t.IsAssignableFrom (typeof (Complex))) {
					size = 16;
					dt = TFDataType.Complex128;
				} else
					throw new ArgumentException ($"The data type {t} is not supported");
				break;
			}
			return (dt, size);
		}

		internal static unsafe object FetchSimple (TFDataType dt, object data)
		{
			switch (dt) {
			case TFDataType.Float:
				return Convert.ToSingle (data);
			case TFDataType.Double:
				return Convert.ToDouble (data);
			case TFDataType.Int32:
				return Convert.ToInt32 (data);
			case TFDataType.UInt8:
				return Convert.ToByte (data);
			case TFDataType.Int16:
				return Convert.ToInt16 (data);
			case TFDataType.Int8:
				return Convert.ToSByte (data);
			case TFDataType.String:
				throw new NotImplementedException ();
			case TFDataType.Int64:
				return Convert.ToInt64 (data);
			case TFDataType.Bool:
				return Convert.ToBoolean (data);
			case TFDataType.UInt16:
				return Convert.ToUInt16 (data);
			case TFDataType.Complex128:
				return (Complex)data;
			default:
				return null;
			}
		}

        internal static TFTensor Create1DTensor(TFDataType dt, object data)
        {
            switch (dt)
            {
                case TFDataType.Float:
                    return new TFTensor(Convert.ToSingle(data));
                case TFDataType.Double:
                    return new TFTensor(Convert.ToDouble(data));
                case TFDataType.Int32:
                    return new TFTensor(Convert.ToInt32(data));
                case TFDataType.UInt8:
                    return new TFTensor(Convert.ToByte(data));
                case TFDataType.Int16:
                    return new TFTensor(Convert.ToInt16(data));
                case TFDataType.Int8:
                    return new TFTensor(Convert.ToSByte(data));
                case TFDataType.String:
                    throw new NotImplementedException();
                case TFDataType.Int64:
                    return new TFTensor(Convert.ToInt64(data));
                case TFDataType.Bool:
                    return new TFTensor(Convert.ToBoolean(data));
                case TFDataType.UInt16:
                    return new TFTensor(Convert.ToUInt16(data));
                case TFDataType.Complex128:
                    return new TFTensor((Complex)data);
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

		//used to create multidementional arrays / tensor with a constant value
		internal static unsafe void Set (Array target, TFDataType dt, long [] shape, int [] idx, int level, object value)
		{
			if (level < shape.Length - 1) {
				for (idx [level] = 0; idx [level] < shape [level]; idx [level]++)
					Set (target, dt, shape, idx, level + 1, value);
			} else {
				for (idx [level] = 0; idx [level] < shape [level]; idx [level]++) {
					switch (dt) {
					case TFDataType.Float:
					case TFDataType.Double:
					case TFDataType.Int32:
					case TFDataType.UInt8:
					case TFDataType.Int16:
					case TFDataType.Int8:
					case TFDataType.Int64:
					case TFDataType.Bool:
					case TFDataType.Complex128:
						target.SetValue (value, idx);
						break;
					case TFDataType.String:
						throw new NotImplementedException ("String decoding not implemented for tensor vecotrs yet");
					default:
						throw new NotImplementedException ();
					}
				}
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
					for (long l = 0; l < top; l++) {

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
			}
			Copy (target, dt, shape, idx, 0, ref data);
		}

		static unsafe void Copy (Array target, TFDataType dt, long [] shape, int [] idx, int level, ref IntPtr data)
		{
			if (level < shape.Length - 1) {
				for (idx [level] = 0; idx [level] < shape [level]; idx [level]++)
					Copy (target, dt, shape, idx, level + 1, ref data);
			} else {
				for (idx [level] = 0; idx [level] < shape [level]; idx [level]++) {
					switch (dt) {
					case TFDataType.Float:
						target.SetValue ((*(float*)data), idx);
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
				var result = Array.CreateInstance (t, Shape [0]);
				FetchFlatArray (result, TensorType, Data);
				return result;
			} else {
				if (jagged) {
					IntPtr data = Data;
					return FetchJaggedArray (t, TensorType, ref data, Shape);
				} else {
					var result = Array.CreateInstance (t, Shape);
					FetchMultiDimensionalArray (result, TensorType, Data, Shape);
					return result;
				}
			}
		}

		/// <summary>
		/// Returns a <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFTensor"/>.
		/// </summary>
		/// <returns>A <see cref="T:System.String"/> that represents the current <see cref="T:TensorFlow.TFTensor"/>.</returns>
		public override string ToString ()
		{
			var n = NumDims;
			if (n == 0)
				return GetValue ().ToString ();

			StringBuilder sb = new StringBuilder ("[");
			for (int i = 0; i < n; i++) {
				sb.Append (TF_Dim (handle, i));
				if (i + 1 < n)
					sb.Append ("x");
			}
			sb.Append ("]");
			return sb.ToString ();
		}

		private static int [] getLength (Array array, bool deep = true, bool max = false)
		{
			// This function gets the length of all dimensions in a multidimensional, jagged, or mixed array.
			// https://github.com/accord-net/framework/blob/b4990721a61f03602d04c12b148215c7eca1b7ac/Sources/Accord.Math/Matrix/Matrix.Construction.cs#L1118
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details

			if (array.Rank == 0)
				return new int [0];

			if (deep && isJagged (array)) {
				if (array.Length == 0)
					return new int [0];

				int [] rest;
				if (!max) {
					rest = getLength (array.GetValue (0) as Array, deep);
				} else {
					// find the max
					rest = getLength (array.GetValue (0) as Array, deep);
					for (int i = 1; i < array.Length; i++) {
						int [] r = getLength (array.GetValue (i) as Array, deep);

						for (int j = 0; j < r.Length; j++) {
							if (r [j] > rest [j])
								rest [j] = r [j];
						}
					}
				}

				return new [] { array.Length }.Concat (rest).ToArray ();
			}

			int [] vector = new int [array.Rank];
			for (int i = 0; i < vector.Length; i++)
				vector [i] = array.GetUpperBound (i) + 1;
			return vector;
		}

		private static Array deepFlatten (Array array)
		{
			// This function converts multidimensional, jagged, or mixed arrays into a single unidimensional array (i.e. flattens the mixed array).
			// https://github.com/accord-net/framework/blob/f78181b82eb6ee6cc7fd10d2a7a55334982c40df/Sources/Accord.Math/Matrix/Matrix.Common.cs#L1625
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
			int totalLength = getTotalLength (array, deep: true);
			var elementType = getInnerMostType (array);
			Array result = Array.CreateInstance (elementType, totalLength);

			int k = 0;
			foreach (object v in enumerateJagged (array))
				result.SetValue (v, k++);
			return result;
		}

		private static IEnumerable enumerateJagged (Array array)
		{
			// This function can enumerate all elements in a multidimensional ,jagged, or mixed array.
			// From https://github.com/accord-net/framework/blob/b4990721a61f03602d04c12b148215c7eca1b7ac/Sources/Accord.Math/Matrix/Jagged.Construction.cs#L1202
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
			var arrays = new Stack<Array> ();
			var counters = new Stack<int> ();

			arrays.Push (array);
			counters.Push (0);
			int depth = 1;

			Array a = array;
			int i = 0;

			while (arrays.Count > 0) {
				if (i >= a.Length) {
					a = arrays.Pop ();
					i = counters.Pop () + 1;
					depth--;
				} else {
					Object e = a.GetValue (i);
					Array next = e as Array;
					if (next == null) {
						yield return e;
						i++;
					} else {
						arrays.Push (a);
						counters.Push (i);
						a = next;
						i = 0;
						depth++;
					}
				}
			}
		}

		private static int getTotalLength (Array array, bool deep = true, bool rectangular = true)
		{
			// From https://github.com/accord-net/framework/blob/b4990721a61f03602d04c12b148215c7eca1b7ac/Sources/Accord.Math/Matrix/Matrix.Construction.cs#L1087
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
			if (deep && isJagged (array.GetType ())) {
				if (rectangular) {
					int rest = getTotalLength (array.GetValue (0) as Array, deep);
					return array.Length * rest;
				} else {
					int sum = 0;
					for (int i = 0; i < array.Length; i++)
						sum += getTotalLength (array.GetValue (i) as Array, deep);
					return sum;
				}
			}

			return array.Length;
		}

		private static bool isJagged (Array array)
		{
			// From https://github.com/accord-net/framework/blob/f78181b82eb6ee6cc7fd10d2a7a55334982c40df/Sources/Accord.Math/Matrix/Matrix.Construction.cs#L1204
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
			if (array.Length == 0)
				return array.Rank == 1;
			return array.Rank == 1 && array.GetValue (0) is Array;
		}

		private static bool isJagged (Type type)
		{
			// From https://github.com/accord-net/framework/blob/eb371fbc540a41c1a711b6ab1ebd49889316e7f7/Sources/Accord.Math/Matrix/Matrix.Common.cs#L84
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
			return type.IsArray && type.GetElementType ().IsArray;
		}

		private static Type getInnerMostType (Array array)
		{
			// From https://github.com/accord-net/framework/blob/eb371fbc540a41c1a711b6ab1ebd49889316e7f7/Sources/Accord.Math/Matrix/Matrix.Common.cs#L95
			// Relicensed under the MIT license by the original author for inclusion in TensorFlowSharp and any derived projects, see the MIT license for details
			Type type = array.GetType ();

			while (type.IsArray)
				type = type.GetElementType ();

			return type;
		}

	}

}	
