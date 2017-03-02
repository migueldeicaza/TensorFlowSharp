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
using System.Text;
using size_t = System.UIntPtr;

namespace TensorFlow
{

	using static NativeMethods;

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
		private string opType;
		private string operName;
		private readonly TFGraph graph;

		public TFOperationDesc (TFGraph graph, string opType, string operName) 
		{
			if (graph == null)
				throw new ArgumentNullException (nameof(graph));

			Handle = TF_NewOperation (graph.Handle, opType, operName);

			this.graph = graph;
			this.opType = opType;
			this.operName = operName;
		}

		protected override void NativeDispose (IntPtr handle) 
		{
			// If you reach this, you never called FinishOperation
			Console.WriteLine ($"TFOperationDescription({opType},{operName} was never turned into an TFOperation");
		}


		public void SetDevice (string device)
		{
			CheckDisposed ();
			if (device == null)
				throw new ArgumentNullException (nameof(device));
			
			TF_SetDevice (Handle, device);
		}


		public void AddInput (TFOutput input)
		{
			CheckDisposed ();

			TF_AddInput (Handle, input);
		}


		public void AddInputs (params TFOutput [] inputs)
		{
			CheckDisposed ();

			// TODO: Check if we can throw an argument exception here.
			if (inputs == null || inputs.Length == 0)
				return;

			TF_AddInputList (Handle, inputs, inputs.Length);
		}


		public void AddControlInput (TFOperation input)
		{
			CheckDisposed ();

			if (input == null)
				throw new ArgumentNullException (nameof(input));

			TF_AddControlInput (Handle, input.Handle);
		}



		public void ColocateWith (TFOperation op)
		{
			CheckDisposed();
			if (op == null)
				throw new ArgumentNullException (nameof(op));
			
			TF_ColocateWith (Handle, op.Handle);
		}


		public void SetAttr (string attrName, string value)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			var bytes = Encoding.UTF8.GetBytes (value);
			var buf = Marshal.AllocHGlobal (bytes.Length + 1);
			Marshal.Copy (bytes, 0, buf, bytes.Length);

			TF_SetAttrString (Handle, attrName, buf, (size_t) bytes.Length);
		}


		public void SetAttr (string attrName, string [] values)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			var n = values.Length;
			var unmanaged = new IntPtr [n];
			var lenghts = new size_t [n];
			for (var i = 0; i < n; i++) {
				var bytes = Encoding.UTF8.GetBytes (values [i]);
				var buf = Marshal.AllocHGlobal (bytes.Length + 1);
				var bc = bytes.Length;

				Marshal.Copy (bytes, 0, buf, bc);
				unmanaged [i] = buf;
				lenghts [i] = (size_t)bc;
			}
			TF_SetAttrStringList (Handle, attrName, unmanaged, lenghts, n);
		}




		public void SetAttr (string attrName, long value)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrInt (Handle, attrName, value);
		}


		public void SetAttr (string attrName, long [] values)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrIntList (Handle, attrName, values, values.Length);
		}



		public void SetAttr (string attrName, float value)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrFloat (Handle, attrName, value);
		}


		public void SetAttr (string attrName, float [] values)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrFloatList (Handle, attrName, values, values.Length);
		}

		public void SetAttr (string attrName, bool value)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrBool (Handle, attrName, (byte)(value ? 1 : 0));
		}


		public void SetAttr (string attrName, bool [] values)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (values == null)
				throw new ArgumentNullException (nameof (values));

			TF_SetAttrBoolList (Handle, attrName, values, values.Length);
		}


		public void SetAttrType (string attrName, TFDataType dataType)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			TF_SetAttrType (Handle, attrName, dataType);
		}



		public void SetAttrType (string attrName, params TFDataType [] dataType)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (dataType == null)
				throw new ArgumentNullException (nameof (dataType));
			TF_SetAttrTypeList (Handle, attrName, dataType, dataType.Length);
		}


		public void SetAttrShape (string attrName, TFShape shape)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (shape == null || shape.dims == null)
				TF_SetAttrShape (Handle, attrName, null, -1);
			else
				TF_SetAttrShape (Handle, attrName, shape.dims, shape.dims.Length);
		}


		public void SetAttrShape (string attrName, TFShape [] shapeList)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (shapeList == null)
				throw new ArgumentNullException (nameof (shapeList));
			var num_shapes = shapeList.Length;
			var num_dims = new int [shapeList.Length];
			unsafe
			{
				var unmanaged = Marshal.AllocHGlobal (sizeof (IntPtr) * num_shapes);
				var ofs = 0;
				for (var i = 0; i < num_shapes; i++) {
					var array = Marshal.AllocHGlobal (sizeof (long) * shapeList [i].dims.Length);
					Marshal.Copy (shapeList [i].dims, 0, array, shapeList [i].dims.Length);
					Marshal.WriteIntPtr (unmanaged, ofs, array);
					ofs += sizeof (IntPtr);
				}
				TF_SetAttrShapeList (Handle, attrName, unmanaged, num_dims, num_shapes);
				ofs = 0;
				for (var i = 0; i < num_shapes; i++) {
					var ptr = Marshal.ReadIntPtr (unmanaged, ofs);
					Marshal.FreeHGlobal (ptr);
					ofs += sizeof (IntPtr);
				}
				Marshal.FreeHGlobal (unmanaged);
			}
		}

		public void SetAttrTensorShapeProto (string attrName, IntPtr proto, size_t protoLen, TFStatus status = null)
		{
			CheckDisposed();
			var cstatus = TFStatus.Setup (status);
			TF_SetAttrTensorShapeProto (Handle, attrName, proto, protoLen, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
		}


		public void SetAttr (string attrName, TFTensor tensor, TFStatus status = null)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (tensor == null)
				throw new ArgumentNullException (nameof(tensor));
			var cstatus = TFStatus.Setup (status);

			TF_SetAttrTensor (Handle, attrName, tensor.Handle, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
		}

		public void SetAttr (string attrName, TFTensor [] tensor, TFStatus status = null)
		{
			CheckDisposed();
			if (attrName == null)
				throw new ArgumentNullException (nameof (attrName));
			if (tensor == null)
				throw new ArgumentNullException (nameof (tensor));
			var cstatus = TFStatus.Setup (status);
			var unmanaged = new IntPtr [tensor.Length];
			for (var i = 0; i < tensor.Length; i++)
				unmanaged [i] = tensor [i].Handle;
			TF_SetAttrTensorList (Handle, attrName, unmanaged, unmanaged.Length, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
		}


		public TFOperation FinishOperation (TFStatus status = null)
		{
			CheckDisposed();
			var cstatus = TFStatus.Setup (status);
			var h = TF_FinishOperation (Handle, cstatus.Handle);
			cstatus.CheckMaybeRaise (status);
			Handle = IntPtr.Zero;
			GC.SuppressFinalize (this);

			return new TFOperation (graph, h);
		}
	}
}
