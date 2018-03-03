using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
	public class ArrayTests
	{
		[Fact]
		public void BasicConstantZerosAndOnes ()
		{
			using (var g = new TFGraph ())
			using (var s = new TFSession (g)) {

				// Test Zeros, Ones for n x n shape
				var o = g.Ones (new TFShape (4, 4));
				Assert.NotNull (o);
				Assert.Equal (o.OutputType, TFDataType.Double);

				var z = g.Zeros (new TFShape (4, 4));
				Assert.NotNull (z);
				Assert.Equal (z.OutputType, TFDataType.Double);

				var r = g.RandomNormal (new TFShape (4, 4));
				Assert.NotNull (r);
				Assert.Equal (r.OutputType, TFDataType.Double);

				var res1 = s.GetRunner ().Run (g.Mul (o, r));
				Assert.NotNull (res1);
				Assert.Equal (res1.TensorType, TFDataType.Double);
				Assert.Equal (res1.NumDims, 2);
				Assert.Equal (res1.Shape [0], 4);
				Assert.Equal (res1.Shape [1], 4);
				Assert.Equal (res1.ToString (), "[4x4]");

				var matval1 = res1.GetValue ();
				Assert.NotNull (matval1);
				Assert.IsType (typeof (double [,]), matval1);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						Assert.NotNull (((double [,])matval1) [i, j]);
					}
				}

				var res2 = s.GetRunner ().Run (g.Mul (g.Mul (o, r), z));
				Assert.NotNull (res2);
				Assert.Equal (res2.TensorType, TFDataType.Double);
				Assert.Equal (res2.NumDims, 2);
				Assert.Equal (res2.Shape [0], 4);
				Assert.Equal (res2.Shape [1], 4);
				Assert.Equal (res2.ToString (), "[4x4]");

				var matval2 = res2.GetValue ();
				Assert.NotNull (matval2);
				Assert.IsType (typeof (double [,]), matval2);

				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						Assert.NotNull (((double [,])matval2) [i, j]);
						Assert.Equal (((double [,])matval2) [i, j], 0.0);
					}
				}
			}
		}

#if false
		[Fact]
		public void BasicConstantsOnSymmetricalShapes ()
		{
			using (var g = new TFGraph ())
			using (var s = new TFSession (g)) {
				//build some test vectors
				var o = g.Ones (new TFShape (4, 4));
				var z = g.Zeros (new TFShape (4, 4));
				var r = g.RandomNormal (new TFShape (4, 4));
				var matval = s.GetRunner ().Run (g.Mul (o, r)).GetValue ();
				var matvalzero = s.GetRunner ().Run (g.Mul (g.Mul (o, r), z)).GetValue ();

				var co = g.Constant (1.0, new TFShape (4, 4), TFDataType.Double);
				var cz = g.Constant (0.0, new TFShape (4, 4), TFDataType.Double);
				var res1 = s.GetRunner ().Run (g.Mul (co, r));

				Assert.NotNull (res1);
				Assert.Equal (res1.TensorType, TFDataType.Double);
				Assert.Equal (res1.NumDims, 2);
				Assert.Equal (res1.Shape [0], 4);
				Assert.Equal (res1.Shape [1], 4);
				Assert.Equal (res1.ToString (), "[4x4]");

				var cmatval1 = res1.GetValue ();
				Assert.NotNull (cmatval1);
				Assert.IsType (typeof (double [,]), cmatval1 );
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						Assert.NotNull (((double [,])cmatval1) [i, j]);
					}
				}

				var cres2 = s.GetRunner ().Run (g.Mul (g.Mul (co, r), cz));

				Assert.NotNull (cres2);
				Assert.Equal (cres2.TensorType, TFDataType.Double);
				Assert.Equal (cres2.NumDims, 2);
				Assert.Equal (cres2.Shape [0], 4);
				Assert.Equal (cres2.Shape [1], 4);
				Assert.Equal (cres2.ToString (), "[4x4]");

				var cmatval2 = cres2.GetValue ();
				Assert.NotNull (cmatval2);
				Assert.IsType (typeof (double [,]), cmatval2);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						Assert.NotNull (((double [,])cmatval2) [i, j]);
						Assert.Equal (((double [,])matvalzero) [i, j], ((double [,])cmatval2) [i, j]);
					}
				}
			}
		}

		[Fact]
		public void BasicConstantsUnSymmetrical ()
		{
			using (var g = new TFGraph ())
			using (var s = new TFSession (g)) {
				var o = g.Ones (new TFShape (4, 3));
				Assert.NotNull (o);
				Assert.Equal (o.OutputType, TFDataType.Double);

				var r = g.RandomNormal (new TFShape (3, 5));
				Assert.NotNull (o);
				Assert.Equal (o.OutputType, TFDataType.Double);

				//expect incompatible shapes
				Assert.Throws<TFException> (() => s.GetRunner ().Run (g.Mul (o, r)));

				var res = s.GetRunner ().Run (g.MatMul (o, r));
				Assert.NotNull (res);
				Assert.Equal (res.TensorType, TFDataType.Double);
				Assert.Equal (res.NumDims, 2);
				Assert.Equal (res.Shape [0], 4);
				Assert.Equal (res.Shape [1], 5);

				double [,] val = (double [,])res.GetValue ();
				Assert.NotNull (val);
				Assert.IsType (typeof (double [,]), val);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 5; j++) {
						Assert.NotNull (((double [,])val) [i, j]);
					}
				}
			}
		}
#endif

        private static IEnumerable<object[]> stackData()
        {
            // Example from https://www.tensorflow.org/api_docs/python/tf/stack

            // 'x' is [1, 4]
            // 'y' is [2, 5]
            // 'z' is [3, 6]

            double[] x = { 1, 4 };
            double[] y = { 2, 5 };
            double[] z = { 3, 6 };

            // stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  // Pack along first dim.
            // stack([x, y, z], axis= 1) => [[1, 2, 3], [4, 5, 6]]

            yield return new object[] { x, y, z, null,  new double[,] { { 1, 4 },
                                                                        { 2, 5 },
                                                                        { 3, 6 } } };

            yield return new object[] { x, y, z, 1, new double[,] { { 1, 2, 3 },
                                                                    { 4, 5, 6 } } };
        }

        [Theory]
        [MemberData(nameof(stackData))]
        public void Should_Stack(double[] x, double[] y, double[] z, int? axis, double[,] expected)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var a = graph.Placeholder(TFDataType.Double, new TFShape(2));
                var b = graph.Placeholder(TFDataType.Double, new TFShape(2));
                var c = graph.Placeholder(TFDataType.Double, new TFShape(2));

                TFOutput r = graph.Stack(new[] { a, b, c }, axis: axis);

                TFTensor[] result = session.Run(new[] { a, b, c }, new TFTensor[] { x, y, z }, new[] { r });

                double[,] actual = (double[,])result[0].GetValue();
                TestUtils.MatrixEqual(expected, actual, precision: 10);
            }
        }

        private static IEnumerable<object[]> rangeData()
        {
            double[] x = { 1, 4 };
            double[] y = { 2, 5 };
            double[] z = { 3, 6 };

            // 'start' is 3
            // 'limit' is 18
            // 'delta' is 3
            //  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

            // 'start' is 3
            // 'limit' is 1
            // 'delta' is -0.5
            //  tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]

            // 'limit' is 5
            //  tf.range(limit) ==> [0, 1, 2, 3, 4]

            yield return new object[] { 3, 18, 3, new int[] { 3, 6, 9, 12, 15 } };
            yield return new object[] { 3, 1, -0.5, new double[] { 3, 2.5, 2, 1.5 } };
            yield return new object[] { 3, 1, -0.5f, new float[] { 3, 2.5f, 2, 1.5f } };
            yield return new object[] { null, 5, null, new int[] { 0, 1, 2, 3, 4 } };
            yield return new object[] { null, 5f, null, new float[] { 0, 1, 2, 3, 4f } };
        }

        [Theory]
        [MemberData(nameof(rangeData))]
        public void Should_Range(object start, object limit, object delta, object expected)
        {
            // Examples from https://www.tensorflow.org/api_docs/python/tf/range

            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                TFOutput tstart = graph.Placeholder(start == null ? TFDataType.Int32 : TensorTypeFromType(start.GetType()));
                TFOutput tlimit = graph.Placeholder(limit == null ? TFDataType.Int32 : TensorTypeFromType(limit.GetType()));
                TFOutput tdelta = graph.Placeholder(delta == null ? TFDataType.Int32 : TensorTypeFromType(delta.GetType()));

                TFTensor[] result;
                if (start == null && delta == null)
                {
                    TFOutput r = graph.Range(tlimit);
                    result = session.Run(new[] { tlimit }, new TFTensor[] { TensorFromObject(limit) }, new[] { r });
                }
                else
                {
                    TFOutput r = graph.Range(tstart, (Nullable<TFOutput>)tlimit, (Nullable<TFOutput>)tdelta);
                    result = session.Run(new[] { tstart, tlimit, tdelta },
                        new TFTensor[] { TensorFromObject(start), TensorFromObject(limit), TensorFromObject(delta) },
                        new[] { r });
                }

                Array actual = (Array)result[0].GetValue();
                TestUtils.MatrixEqual((Array)expected, actual, precision: 10);
            }
        }


		private static IEnumerable<object []> transposeData ()
		{
			yield return new object [] { new double [,] { { 1, 2 },
														  { 3, 4 } }};
			yield return new object [] { new double [,] { { 1, 2, 3 },
														  { 4, 5, 6} }};
			yield return new object [] { new double [,] { { 1 },
														  { 3 } }};
			yield return new object [] { new double [,] { { 1, 3 } }};
		}

		[Theory]
		[MemberData (nameof (transposeData))]
		public void nShould_Transpose (double [,] x)
		{
			using (var graph = new TFGraph ())
			using (var session = new TFSession (graph)) {
				TFOutput a = graph.Placeholder (TFDataType.Double, new TFShape (2));

				TFOutput r = graph.Transpose (a);

				TFTensor [] result = session.Run (new [] { a }, new TFTensor [] { x }, new [] { r });

				double [,] actual = (double [,])result [0].GetValue ();
				double [,] expected = new double [x.GetLength (1), x.GetLength (0)];
				for (int i = 0; i < expected.GetLength(0); i++) {
					for (int j = 0; j < expected.GetLength(1); j++) {
						expected [i, j] = x [j, i];
					}
				}

				TestUtils.MatrixEqual (expected, actual, precision: 10);
			}
		}



		public static TFDataType TensorTypeFromType(Type type)
        {
            if (type == typeof(float))
                return TFDataType.Float;
            if (type == typeof(double))
                return TFDataType.Double;
            if (type == typeof(int))
                return TFDataType.Int32;
            if (type == typeof(byte))
                return TFDataType.UInt8;
            if (type == typeof(short))
                return TFDataType.Int16;
            if (type == typeof(sbyte))
                return TFDataType.Int8;
            if (type == typeof(String))
                return TFDataType.String;
            if (type == typeof(bool))
                return TFDataType.Bool;
            if (type == typeof(long))
                return TFDataType.Int64;
            if (type == typeof(ushort))
                return TFDataType.UInt16;
            if (type == typeof(Complex))
                return TFDataType.Complex128;

            throw new ArgumentOutOfRangeException("type");
        }

        public static TFTensor TensorFromObject(object obj)
        {
            Type type = obj.GetType();
            if (type == typeof(float))
                return new TFTensor((float)obj);
            if (type == typeof(double))
                return new TFTensor((double)obj);
            if (type == typeof(int))
                return new TFTensor((int)obj);
            if (type == typeof(byte))
                return new TFTensor((byte)obj);
            if (type == typeof(short))
                return new TFTensor((short)obj);
            if (type == typeof(sbyte))
                return new TFTensor((sbyte)obj);
            if (type == typeof(String))
                throw new NotImplementedException();
            if (type == typeof(bool))
                return new TFTensor((bool)obj);
            if (type == typeof(long))
                return new TFTensor((long)obj);
            if (type == typeof(ushort))
                return new TFTensor((ushort)obj);
            if (type == typeof(Complex))
                return new TFTensor((Complex)obj);

            throw new ArgumentOutOfRangeException("type");
        }
    }
}
