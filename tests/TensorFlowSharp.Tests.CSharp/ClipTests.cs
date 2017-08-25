using System.Collections.Generic;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class ClipTests
    {
        private static IEnumerable<object[]> clipByValueData()
        {
            yield return new object[]
            {
                new double[,]
                {
                    { 1, 2, 0 },
                    { 4, -1, 7 }
                },

                1.0, 6.0,

                new double[,]
                {
                    { 1, 2, 1 },
                    { 4, 1, 6 }
                }
            };

            yield return new object[]
            {
                new double[,]
                {
                    { -9, double.PositiveInfinity, double.NegativeInfinity },
                    { 4, double.PositiveInfinity, 7 }
                },

                -2, 42,

                new double[,]
                {
                    { -2, 42, -2 },
                    { 4, 42, 7 }
                }
            };
        }

        [Theory]
        [MemberData(nameof(clipByValueData))]
        public void Should_ClipByValue(double[,] m, double min, double max, double[,] expected)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var matrix = graph.Placeholder(TFDataType.Double);
                var clip_min = graph.Placeholder(TFDataType.Double);
                var clip_max = graph.Const(new TFTensor(max));

                TFOutput y = graph.ClipByValue(matrix, clip_min, clip_max);

                TFTensor[] result = session.Run(new[] { matrix, clip_min }, new TFTensor[] { m, min }, new[] { y });

                double[,] actual = (double[,])result[0].GetValue();
                Assert.Equal(expected, actual);
            }
        }






        private static IEnumerable<object[]> clipByNormData()
        {
            yield return new object[]
            {
                new double[,]
                {
                    { 1, 2, 0 },
                    { 4, -1, 7 }
                },

                1.0, 0,

                new double[,]
                {
                    { 0.24253562503633297, 0.89442719099991586, 0 },
                    { 0.97014250014533188, -0.44721359549995793, 1 }
                }
            };

            yield return new object[]
            {
                new double[,]
                {
                    { -9, 100, 0.1 },
                    { 4, 0.4, 7 }
                },

                -2, 1,

                new double[,]
                {
                    { -9, 100, 0.1 },
                    { 4, 0.4, 7 }
                }
            };

            yield return new object[]
{
                new double[,]
                {
                    { 1e-10, 1e-5, 1e-3 },
                    { 1e-2, 0.0, 7.0 }
                },

                -2, 2,

                null
            };

            yield return new object[]
            {
                new double[,]
                {
                    { -9, double.PositiveInfinity, double.NegativeInfinity },
                    { 4, double.PositiveInfinity, 7 }
                },

                -2, -1,

                new double[,]
                {
                    { -9, double.PositiveInfinity, double.NegativeInfinity },
                    { 4, double.PositiveInfinity, 7 }
                }
            };
        }

        [Theory]
        [MemberData(nameof(clipByNormData))]
        public void Should_ClipByNorm(double[,] m, double norm, int axis, double[,] expected)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var matrix = graph.Placeholder(TFDataType.Double);
                var clip_norm = graph.Placeholder(TFDataType.Double);
                var a = graph.Const(new TFTensor(axis));

                TFOutput y = graph.ClipByNorm(matrix, clip_norm, a);

                if (expected != null)
                {
                    TFTensor[] result = session.Run(new[] { matrix, clip_norm }, new TFTensor[] { m, norm }, new[] { y });

                    double[,] actual = (double[,])result[0].GetValue();
                    TestUtils.MatrixEqual(expected, actual, precision: 10);
                }
                else
                {
                    Assert.Throws<TFException>(() => session.Run(new[] { matrix, clip_norm }, new TFTensor[] { m, norm }, new[] { y }));
                }
            }
        }










        private static IEnumerable<object[]> clipByAverageNormData()
        {
            yield return new object[]
            {
                new double[,]
                {
                    { 1, 2, 0 },
                    { 4, -1, 7 }
                },

                1.0, 0,

                new double[,]
                {
                    { 0.24253562503633297, 0.89442719099991586, 0 },
                    { 0.97014250014533188, -0.44721359549995793, 1 }
                }
            };

            yield return new object[]
            {
                new double[,]
                {
                    { -9, 100, 0.1 },
                    { 4, 0.4, 7 }
                },

                -2, 1,

                new double[,]
                {
                    { -9, 100, 0.1 },
                    { 4, 0.4, 7 }
                }
            };

            yield return new object[]
{
                new double[,]
                {
                    { 1e-10, 1e-5, 1e-3 },
                    { 1e-2, 0.0, 7.0 }
                },

                -2, 2,

                null
            };

            yield return new object[]
            {
                new double[,]
                {
                    { -9, double.PositiveInfinity, double.NegativeInfinity },
                    { 4, double.PositiveInfinity, 7 }
                },

                -2, -1,

                new double[,]
                {
                    { -9, double.PositiveInfinity, double.NegativeInfinity },
                    { 4, double.PositiveInfinity, 7 }
                }
            };
        }

        [Theory]
        [MemberData(nameof(clipByAverageNormData))]
        public void Should_ClipByAverageNorm(double[,] m, double norm, int axis, double[,] expected)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var matrix = graph.Placeholder(TFDataType.Double);
                var clip_norm = graph.Placeholder(TFDataType.Double);
                var a = graph.Const(new TFTensor(axis));

                TFOutput y = graph.ClipByNorm(matrix, clip_norm, a);

                if (expected != null)
                {
                    TFTensor[] result = session.Run(new[] { matrix, clip_norm }, new TFTensor[] { m, norm }, new[] { y });

                    double[,] actual = (double[,])result[0].GetValue();
                    TestUtils.MatrixEqual(expected, actual, precision: 10);
                }
                else
                {
                    Assert.Throws<TFException>(() => session.Run(new[] { matrix, clip_norm }, new TFTensor[] { m, norm }, new[] { y }));
                }
            }
        }

    }
}
