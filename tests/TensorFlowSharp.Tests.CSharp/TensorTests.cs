using System;
using System.Collections.Generic;
using System.Numerics;
using TensorFlow;
using System.Text;
using Xunit;
using Learn.Mnist;

namespace TensorFlowSharp.Tests.CSharp
{
	public class TensorTests
	{
		private static IEnumerable<object []> jaggedData ()
		{
			yield return new object [] {
				new double [][] { new [] { 1.0, 2.0 }, new [] { 3.0, 4.0 } },
				new double [,] { { 1.0, 2.0}, { 3.0, 4.0 } },
				true
			};

			yield return new object [] {
				new double [][] { new [] { 1.0, 2.0 }, new [] { 1.0, 4.0 } },
				new double [,] { { 1.0, 2.0}, { 3.0, 4.0 } },
				false
			};

			yield return new object [] {
				new double [][][] { new [] { new [] { 1.0 }, new[] { 2.0 } }, new [] { new [] { 3.0 }, new [] { 4.0 } } },
				new double [,,] { { { 1.0 }, { 2.0 } }, { { 3.0 }, { 4.0 } } },
				true
			};

			yield return new object [] {
				new double [][][] { new [] { new [] { 1.0 }, new[] { 2.0 } }, new [] { new [] { 1.0 }, new [] { 4.0 } } },
				new double [,,] { { { 1.0 }, { 2.0 } }, { { 3.0 }, { 4.0 } } },
				false
			};
		}


		[Theory]
		[MemberData (nameof (jaggedData))]
		public void Should_MultidimensionalAndJaggedBeEqual (Array jagged, Array multidimensional, bool expected)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				var tjagged = graph.Const (new TFTensor (jagged));
				var tmultidimensional = graph.Const (new TFTensor (multidimensional));

				TFOutput y = graph.Equal (tjagged, tmultidimensional);
				TFOutput r;
				if (multidimensional.Rank == 2)
					r = graph.All (y, graph.Const (new [] { 0, 1 }));
				else if (multidimensional.Rank == 3)
					r = graph.All (y, graph.Const (new [] { 0, 1, 2 }));
				else
					throw new System.Exception ("If you want to test Ranks > 3 please handle this extra case manually.");

				TFTensor [] result = session.Run (new TFOutput [] { }, new TFTensor [] { }, new [] { r });

				bool actual = (bool)result [0].GetValue ();
				Assert.Equal (expected, actual);
			}
		}

		[Fact]
		public void StringTestWithMultiDimStringTensorAsInputOutput ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph))
			{
				var W = graph.Placeholder (TFDataType.String, new TFShape (-1, 2));
				var identityW = graph.Identity (W);

				var dataW = new string [,] { { "This is fine.", "That's ok." }, { "This is fine.", "That's ok." } };
				var bytes = new byte [2 * 2] [];
				bytes [0] = Encoding.UTF8.GetBytes (dataW [0, 0]);
				bytes [1] = Encoding.UTF8.GetBytes (dataW [0, 1]);
				bytes [2] = Encoding.UTF8.GetBytes (dataW [1, 0]);
				bytes [3] = Encoding.UTF8.GetBytes (dataW [1, 1]);
				var tensorW = TFTensor.CreateString (bytes, new TFShape (2, 2));

				var outputTensor = session.Run (new TFOutput [] { W }, new TFTensor [] { tensorW }, new [] { identityW });

				var outputW = TFTensor.DecodeMultiDimensionString (outputTensor [0]);
				Assert.Equal (dataW [0, 0], Encoding.UTF8.GetString (outputW [0]));
				Assert.Equal (dataW [0, 1], Encoding.UTF8.GetString (outputW [1]));
				Assert.Equal (dataW [1, 0], Encoding.UTF8.GetString (outputW [2]));
				Assert.Equal (dataW [1, 1], Encoding.UTF8.GetString (outputW [3]));
			}
		}

		[Fact]
		public void StringTestWithMultiDimStringTensorAsInputAndScalarStringAsOutput ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph))
			{
				var X = graph.Placeholder (TFDataType.String, new TFShape (-1));
				var delimiter = graph.Const (TFTensor.CreateString (Encoding.UTF8.GetBytes ("/")));
				var indices = graph.Const (0);
				var Y = graph.ReduceJoin (graph.StringSplit (X, delimiter).values, indices, separator: " ");

				var dataX = new string [] { "Thank/you/very/much!.", "I/am/grateful/to/you.", "So/nice/of/you." };
				var bytes = new byte [dataX.Length] [];
				bytes [0] = Encoding.UTF8.GetBytes (dataX [0]);
				bytes [1] = Encoding.UTF8.GetBytes (dataX [1]);
				bytes [2] = Encoding.UTF8.GetBytes (dataX [2]);
				var tensorX = TFTensor.CreateString (bytes, new TFShape (3));

				var outputTensors = session.Run (new TFOutput [] { X }, new TFTensor [] { tensorX }, new [] { Y });

				var outputY = Encoding.UTF8.GetString (TFTensor.DecodeString (outputTensors [0]));
				Assert.Equal (string.Join (" ", dataX).Replace ("/", " "), outputY);
			}
		}

		[Fact (Skip = "Disabled because it requires GPUs and need to set numGPUs to available GPUs on system." +
			" It has been tested on GPU machine with 4 GPUs and it passed there.")]
		public void DevicePlacementTest ()
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph))
			{
				var X = graph.Placeholder (TFDataType.Float, new TFShape (-1, 784));
				var Y = graph.Placeholder (TFDataType.Float, new TFShape (-1, 10));

				int numGPUs = 4;
				var Xs = graph.Split (graph.Const (0), X, numGPUs);
				var Ys = graph.Split (graph.Const (0), Y, numGPUs);
				var products = new TFOutput [numGPUs];
				for (int i = 0; i < numGPUs; i++)
				{
					using (var device = graph.WithDevice ("/device:GPU:" + i))
					{
						var W = graph.Constant (0.1f, new TFShape (784, 500), TFDataType.Float);
						var b = graph.Constant (0.1f, new TFShape (500), TFDataType.Float);
						products [i] = graph.Add (graph.MatMul (Xs [i], W), b);
					}
				}
				var stacked = graph.Concat (graph.Const (0), products);
				Mnist mnist = new Mnist ();
				mnist.ReadDataSets ("/tmp");
				int batchSize = 1000;
				for (int i = 0; i < 100; i++)
				{
					var reader = mnist.GetTrainReader ();
					(var trainX, var trainY) = reader.NextBatch (batchSize);
					var outputTensors = session.Run (new TFOutput [] { X }, new TFTensor [] { new TFTensor (trainX) }, new TFOutput [] { stacked });
					Assert.Equal (1000, outputTensors [0].Shape [0]);
					Assert.Equal (500, outputTensors [0].Shape [1]);
				}

			}
		}

		[Fact]
		public void ConstructBoolTensor ()
		{
			bool value = true;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Bool, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (bool), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructByteTensor ()
		{
			byte value = 123;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.UInt8, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (byte), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructSignedByteTensor ()
		{
			sbyte value = 123;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Int8, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (sbyte), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructShortTensor ()
		{
			short value = 123;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Int16, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (short), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructUnsignedShortTensor ()
		{
			ushort value = 123;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.UInt16, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (ushort), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructIntTensor ()
		{
			int value = 123;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Int32, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (int), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructLongTensor ()
		{
			long value = 123L;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Int64, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (long), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructComplexTensor ()
		{
			Complex value = new Complex (1, 2);
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Complex128, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal (16u, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructFloatTensor ()
		{
			float value = 123.456f;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Float, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (float), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructDoubleTensor ()
		{
			double value = 123.456;
			using (var tensor = new TFTensor (value))
			{
				Assert.Equal (TFDataType.Double, tensor.TensorType);
				Assert.Equal (0, tensor.NumDims);
				Assert.Equal (new long [0], tensor.Shape);
				Assert.Equal ((uint)sizeof (double), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (value, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructBoolArrayTensor ()
		{
			var array = new [] { true, false };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Bool, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (bool) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructByteArrayTensor ()
		{
			var array = new byte [] { 123, 234 };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.UInt8, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (byte) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructSignedByteArrayTensor ()
		{
			var array = new sbyte [] { 123, -123 };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Int8, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (sbyte) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructShortArrayTensor ()
		{
			var array = new short [] { 123, 234 };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Int16, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (short) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructUnsignedShortArrayTensor ()
		{
			var array = new ushort [] { 123, 234 };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.UInt16, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (ushort) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructIntArrayTensor ()
		{
			var array = new [] { 123, 234 };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Int32, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (int) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructMultiDimIntArrayTensor ()
		{
			var array = new [,] { { 123, 456 } };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Int32, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.GetLength (0), tensor.GetTensorDimension (0));
				Assert.Equal (array.GetLength (1), tensor.GetTensorDimension (1));
				Assert.Equal (new long [] { array.GetLength (0), array.GetLength (1) }, tensor.Shape);
				Assert.Equal ((uint)sizeof (int) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructJaggedIntArrayTensor ()
		{
			var jagged = new [] { new [] { 123, 456 } };
			var array = new [,] { { 123, 456 } };
			using (var tensor = new TFTensor (jagged))
			{
				Assert.Equal (TFDataType.Int32, tensor.TensorType);
				Assert.Equal (2, tensor.NumDims);
				Assert.Equal (array.GetLength (0), tensor.GetTensorDimension (0));
				Assert.Equal (array.GetLength (1), tensor.GetTensorDimension (1));
				Assert.Equal (new long [] { array.GetLength (0), array.GetLength (1) }, tensor.Shape);
				Assert.Equal ((uint)sizeof (int) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructLongArrayTensor ()
		{
			var array = new [] { 123L, 234L };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Int64, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (long) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstrucComplexArrayTensor ()
		{
			var array = new [] { new Complex (1, 2), new Complex (2, -1) };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Complex128, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal (16u * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructFloatArrayTensor ()
		{
			var array = new [] { 123.456f, 234.567f };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Float, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (float) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void ConstructDoubleArrayTensor ()
		{
			var array = new [] { 123.456, 234.567 };
			using (var tensor = new TFTensor (array))
			{
				Assert.Equal (TFDataType.Double, tensor.TensorType);
				Assert.Equal (array.Rank, tensor.NumDims);
				Assert.Equal (array.Length, tensor.GetTensorDimension (0));
				Assert.Equal (new long [] { array.Length }, tensor.Shape);
				Assert.Equal ((uint)sizeof (double) * array.Length, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (array, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetArrayTensor ()
		{
			using (var tensor = new TFTensor (new [] { 123, 456 }))
			{
				tensor.SetValue (new [] { 234, 567 });
				Assert.Equal ((uint)sizeof (int) * 2, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (new [] { 234, 567 }, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetMultiDimArrayTensor ()
		{
			using (var tensor = new TFTensor (new [,] { { 123, 456 } }))
			{
				tensor.SetValue (new [,] { { 234, 567 } });
				Assert.Equal ((uint)sizeof (int) * 2, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (new [,] { { 234, 567 } }, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetMultiDimArrayTensorWithJagged ()
		{
			using (var tensor = new TFTensor (new [,] { { 123, 456 } }))
			{
				tensor.SetValue (new [] { new [] { 234, 567 } });
				Assert.Equal ((uint)sizeof (int) * 2, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (new [,] { { 234, 567 } }, tensor.GetValue ());

			}
		}

		[Fact]
		public void SetJaggedArrayTensor ()
		{
			using (var tensor = new TFTensor (new [] { new [] { 123, 456 } }))
			{
				tensor.SetValue (new [] { new [] { 234, 567 } });
				Assert.Equal ((uint)sizeof (int) * 2, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (new [,] { { 234, 567 } }, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetBoolTensor ()
		{
			using (var tensor = new TFTensor (true))
			{
				tensor.SetValue (false);
				Assert.Equal ((uint)sizeof (bool), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (false, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetByteTensor ()
		{
			using (var tensor = new TFTensor ((byte)123))
			{

				tensor.SetValue ((byte)234);
				Assert.Equal ((uint)sizeof (byte), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal ((byte)234, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetSignedByteTensor ()
		{
			using (var tensor = new TFTensor ((sbyte)123))
			{
				tensor.SetValue ((sbyte)-123);
				Assert.Equal ((uint)sizeof (sbyte), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal ((sbyte)-123, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetShortTensor ()
		{
			using (var tensor = new TFTensor ((short)123))
			{
				tensor.SetValue ((short)234);
				Assert.Equal ((uint)sizeof (short), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal ((short)234, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetUnsignedShortTensor ()
		{
			using (var tensor = new TFTensor ((ushort)123))
			{
				tensor.SetValue ((ushort)234);
				Assert.Equal ((uint)sizeof (ushort), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal ((ushort)234, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetIntTensor ()
		{
			using (var tensor = new TFTensor (123))
			{
				tensor.SetValue (234);
				Assert.Equal ((uint)sizeof (int), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (234, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetLongTensor ()
		{
			using (var tensor = new TFTensor (123L))
			{
				tensor.SetValue (234L);
				Assert.Equal ((uint)sizeof (long), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (234L, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetComplexTensor ()
		{
			using (var tensor = new TFTensor (new Complex (1, 2)))
			{
				tensor.SetValue (new Complex (2, -1));
				Assert.Equal ((uint)16, tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (new Complex (2, -1), tensor.GetValue ());
			}
		}

		[Fact]
		public void SetFloatTensor ()
		{
			using (var tensor = new TFTensor (123.456f))
			{

				tensor.SetValue (234.567f);
				Assert.Equal ((uint)sizeof (float), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (234.567f, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetDoubleTensor ()
		{
			using (var tensor = new TFTensor (123.456))
			{
				tensor.SetValue (234.567);
				Assert.Equal ((uint)sizeof (double), tensor.TensorByteSize.ToUInt32 ());
				Assert.Equal (234.567, tensor.GetValue ());
			}
		}

		[Fact]
		public void SetTensorWithWrongType ()
		{
			using (var tensor = new TFTensor (123))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.SetValue ((ushort)234));
				Assert.Equal ("The tensor is of type Int32, not UInt16", exception.Message);
			}
		}

		[Fact]
		public void SetArrayTensorWithSimple ()
		{
			using (var tensor = new TFTensor (new [] { 123 }))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.SetValue (234));
				Assert.Equal ("This tensor is an array tensor, not a simple tensor", exception.Message);
			}
		}

		[Fact]
		public void SetArrayTensorWithWrongDimensions ()
		{
			using (var tensor = new TFTensor (new [,] { { 123 } }))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.SetValue (new [] { 234 }));
				Assert.Equal ("This tensor has 2 dimensions, the given array has 1", exception.Message);
			}
		}

		[Fact]
		public void SetArrayTensorWithWrongLength ()
		{
			using (var tensor = new TFTensor (new [,] { { 123 } }))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.SetValue (new [,] { { 234, 567 } }));
				Assert.Equal ("This tensor has shape [1,1], the given array has shape [1,2]", exception.Message);
			}
		}

		[Fact]
		public void SetJaggedArrayTensorWithWrongDimensions ()
		{
			using (var tensor = new TFTensor (new [] { new [] { 123 } }))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.SetValue (new [] { new [] { new [] { 234 } } }));
				Assert.Equal ("This tensor has 2 dimensions, the given array has 3", exception.Message);
			}
		}

		[Fact]
		public void SetJaggedArrayTensorWithWrongLength ()
		{
			using (var tensor = new TFTensor (new [] { new [] { 123 } }))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.SetValue (new [] { new [] { 234, 567 } }));
				Assert.Equal ("This tensor has shape [1,1], given array has shape [1,2]", exception.Message);
			}
		}

		public static IEnumerable<object []> GetArrayValueInPlaceData => new List<object []>
		{
			new [] { new [] { 123 } },
			new [] { new [,] { { 123, 456 } } },
			new [] { new [,,] { { { 123, 456, 789 } } } },
			new [] { new [,] { { 123, 456 }, { 789, 012 } } }
		};

		[Theory]
		[MemberData(nameof(GetArrayValueInPlaceData))]
		public void GetArrayValueInPlace (Array array)
		{
			using (var tensor = new TFTensor (array))
			{
				var type = array.GetType ().GetElementType ();
				var value = Array.CreateInstance (type, tensor.Shape);
				Assert.NotEqual (array, value);

				tensor.GetValue (value);

				Assert.Equal (array, value);
			}
		}

		private static IEnumerable<object []> checkShapeData ()
		{
			yield return new object [] { new [] { 123 }, new [] { 234 }, null };
			yield return new object [] { new [] { 123 }, new [,] { { 234 } }, "This tensor has 1 dimensions, the given array has 2" };
			yield return new object [] { new [] { 123 }, new [] { 234, 567 }, "This tensor has shape [1], the given array has shape [2]" };

			yield return new object [] { new [] { 123, 456 }, new [] { 234, 567 }, null };
			yield return new object [] { new [] { 123, 456 }, new [,] { { 234, 567 } }, "This tensor has 1 dimensions, the given array has 2" };
			yield return new object [] { new [] { 123, 456 }, new [] { 234 }, "This tensor has shape [2], the given array has shape [1]" };

			yield return new object [] { new [,] { { 123 } }, new [,] { { 234 } }, null };
			yield return new object [] { new [,] { { 123 } }, new [] { 234 }, "This tensor has 2 dimensions, the given array has 1" };
			yield return new object [] { new [,] { { 123 } }, new [,] { { 234, 567 } }, "This tensor has shape [1,1], the given array has shape [1,2]" };

			yield return new object [] { new [,] { { 123, 456 }, { 789, 012 } }, new [,] { { 234, 567 }, { 890, 123 } }, null };
			yield return new object [] { new [,] { { 123, 456 }, { 789, 012 } }, new [] { 234, 567 }, "This tensor has 2 dimensions, the given array has 1" };
			yield return new object [] { new [,] { { 123, 456 }, { 789, 012 } }, new [,] { { 234 }, { 890 } }, "This tensor has shape [2,2], the given array has shape [2,1]" };
		}

		[Theory]
		[MemberData (nameof (checkShapeData))]
		public void CheckShape (Array tensorArray, Array checkArray, String expected)
		{
			using (var tensor = new TFTensor (tensorArray))
				AssertCheck (() => tensor.CheckShape (checkArray), expected);
		}

		[Fact]
		public void CheckShapeOnSimpleTensor ()
		{
			using (var tensor = new TFTensor (123))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.CheckShape (new [] { 123 }));
				Assert.Equal ("This tensor has 0 dimensions, the given array has 1", exception.Message);
			}
		}

		private static IEnumerable<object []> checkSimpleDataTypeData ()
		{
			yield return new object [] { typeof (int), null };
			yield return new object [] { typeof (short), "The tensor is of type Int32, not Int16" };
			yield return new object [] { typeof (byte), "The tensor is of type Int32, not UInt8" };
		}

		[Theory]
		[MemberData (nameof (checkSimpleDataTypeData))]
		public void CheckSimpleDataType (Type type, String expected)
		{
			using (var tensor = new TFTensor (123))
				AssertCheck(() => tensor.CheckSimpleDataType (type), expected);
		}

		[Fact]
		public void CheckSimpleDataTypeWithArray ()
		{
			using (var tensor = new TFTensor (123))
			{
				var exception = Assert.Throws<InvalidOperationException> (() => tensor.CheckSimpleDataType (typeof (Array)));
				Assert.Equal ("An array is not a simple type, use CheckDataTypeAndSize(Type type, long length)", exception.Message);
			}
		}

		[Fact]
		public void CheckSimpleDataTypeOnArrayTensor ()
		{
			using (var tensor = new TFTensor (new [] { 123 }))
			{
				var exception = Assert.Throws<ArgumentException> (() => tensor.CheckSimpleDataType (typeof (int)));
				Assert.Equal ("This tensor is an array tensor, not a simple tensor", exception.Message);
			}
		}

		private static IEnumerable<object []> checkDataTypeAndSizeData ()
		{
			yield return new object [] { new TFTensor (123), typeof (int), 1, null };
			yield return new object [] { new TFTensor (123), typeof (short), 1, "The tensor is of type Int32, not Int16" };
			yield return new object [] { new TFTensor (123), typeof (byte), 1, "The tensor is of type Int32, not UInt8" };
			yield return new object [] { new TFTensor (new [] { 123 }), typeof (int), 1, null };
			yield return new object [] { new TFTensor (new [] { 123, 456 }), typeof (int), 2, null };
			yield return new object [] { new TFTensor (new [] { 123, 456, 789 }), typeof (int), 2, "The tensor is of size 12, not 8" };
			yield return new object [] { new TFTensor (new short [] { 123 }), typeof (ushort), 1, "The tensor is of type Int16, not UInt16" };
		}

		[Theory]
		[MemberData (nameof (checkDataTypeAndSizeData))]
		public void CheckDataTypeAndSize (TFTensor tensor, Type type, long length, String expected)
		{
			using (tensor)
				AssertCheck(() => tensor.CheckDataTypeAndSize(type, length), expected);
		}

		public void AssertCheck (Action check, string expected)
		{
			if (expected == null)
			{
				check ();
			}
			else
			{
				var exception = Assert.Throws<ArgumentException> (check);
				Assert.Equal (expected, exception.Message);
			}
		}
	}
}
