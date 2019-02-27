using System;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class TFOutputTests
    {
        public class Construction
        {
            [Fact]
            public void OutputConstructedFromOperationWillGetTheOperationsHandle()
            {
                using (var graph = new TFGraph())
                {
                    var op = graph.NoOp();
                    var output = new TFOutput(op, 0);
                    Assert.Equal(op.Handle, output.LLOperation);
                }
            }
        }

        public class Operators
        {
            public class TFOutputAndTFOutput
            {
                [Theory]
                [InlineData(0, 0, 0)]
                [InlineData(1, 1, 2)]
                [InlineData(1.5, 2.75, 4.25)]
                public void AdditionOperatorYieldsAddedResults(double a, double b, double expected) => RunOperation((x, y) => x + y, a, b, expected);

                [Theory]
                [InlineData(0, 0, 0)]
                [InlineData(1, 1, 0)]
                [InlineData(1.5, 2.75, -1.25)]
                public void SubtractionOperatorYieldsSubtractedResults(double a, double b, double expected) => RunOperation((x, y) => x - y, a, b, expected);

                [Theory]
                [InlineData(0, 1, 0)]
                [InlineData(1, 1, 1)]
                [InlineData(6.5, -2, -3.25)]
                public void DivisionOperatorYieldsDividedResults(double a, double b, double expected) => RunOperation((x, y) => x / y, a, b, expected);

                [Theory]
                [InlineData(0, 0, 0)]
                [InlineData(1, 1, 1)]
                [InlineData(-1.5, 2.75, -4.125)]
                public void MultiplicationOperatorWithScalarYieldsMultipliedResultsAsScalar(double a, double b, double expected) => RunOperation((x, y) => x * y, a, b, expected);

                [Fact]
                public void MultiplicationOfMatricesYieldsMatrixMultiplicationResult()
                {
                    using(var graph = new TFGraph())
                    {
                        var a = graph.Const(new TFTensor(new[,] { { 1.0, 0.0 }, { 0.0, 2.0 } }));
                        var b = graph.Const(new TFTensor(new[,] { { 2.0, 1.0 }, { 3.0, 4.0 } }));

                        var output = a * b;
                        using(var session = new TFSession(graph))
                        {
                            var result = (double[,])session.GetRunner().Run(output).GetValue();
                            Assert.Equal(2.0, result[0, 0]);
                            Assert.Equal(1.0, result[0, 1]);
                            Assert.Equal(6.0, result[1, 0]);
                            Assert.Equal(8.0, result[1, 1]);
                        }
                    }
                }

                [Fact]
                public void MultiplicationOfVectorsYieldsElementwiseMultiplicationResult()
                {
                    using (var graph = new TFGraph())
                    {
                        var a = graph.Const(new TFTensor(new[] { 2.0, 3.0 }));
                        var b = graph.Const(new TFTensor(new[] { 0.5, 4.0 }));

                        var output = a * b;
                        using (var session = new TFSession(graph))
                        {
                            var result = (double[])session.GetRunner().Run(output).GetValue();
                            Assert.Equal(1.0, result[0]);
                            Assert.Equal(12.0, result[1]);
                        }
                    }
                }

                private static void RunOperation(Func<TFOutput, TFOutput, TFOutput> operation, double a, double b, double expected)
                {
                    using (var graph = new TFGraph())
                    {
                        var aop = graph.Const(a);
                        var bop = graph.Const(b);
                        var sut = operation(aop, bop);
                        AssertExpectedOutcome(graph, sut, expected);
                    }
                }
            }

            public class TFOutputAndValue
            {

                [Theory]
                [InlineData(0, 0, 0)]
                [InlineData(1, 1, 2)]
                [InlineData(1.5, 2.75, 4.25)]
                public void AdditionWithFloatIsEquivalentToAdditionWithConstant(float a, float b, float expected) =>
                    AssertSymmetricOperation((x, y) => x + y, (x, y) => x + y, val => new TFTensor(val), a, b, expected);

                [Theory]
                [InlineData(0, 0, 0)]
                [InlineData(1, 1, 0)]
                [InlineData(1.5, 2.75, -1.25)]
                public void SubtractionWithFloatIsEquivalentToSubtractionWithConstant(float a, float b, float expected) => 
                    AssertSymmetricOperation((x, y) => x - y, (x, y) => x - y, val => new TFTensor(val), a, b, expected);

                [Theory]
                [InlineData(0, 1, 0)]
                [InlineData(1, 1, 1)]
                [InlineData(6.5, -2, -3.25)]
                public void DivisionWithFloatIsEquivalentToDivisionWithConstant(float a, float b, float expected) =>
                    AssertSymmetricOperation((x, y) => x / y, (x, y) => x / y, val => new TFTensor(val), a, b, expected);

                [Theory]
                [InlineData(0, 0, 0)]
                [InlineData(1, 1, 1)]
                [InlineData(-1.5, 2.75, -4.125)]
                public void MultiplicationWithFloatIsEquivalentToMultiplicationWithConstant(float a, float b, float expected) =>
                    AssertSymmetricOperation((x, y) => x * y, (x, y) => x * y, val => new TFTensor(val), a, b, expected);




                private static void AssertSymmetricOperation<T>(
                    Func<TFOutput, T, TFOutput> op,
                    Func<T, TFOutput, TFOutput> opRev, 
                    Func<T, TFTensor> tensorFactory,
                    T a, 
                    T b, 
                    T expected)
                {
                    using (var graph = new TFGraph())
                    {
                        var aop = graph.Const(tensorFactory(a));
                        var bop = graph.Const(tensorFactory(b));

                        AssertExpectedOutcome(graph, op(aop, b), expected);
                        AssertExpectedOutcome(graph, opRev(a, bop), expected);
                    }
                }
            }

            public static void AssertExpectedOutcome<T>(TFGraph graph, TFOutput output, T expected)
            {
                using (var session = new TFSession(graph))
                {
                    var result = session.GetRunner().Run(output);

                    Assert.Equal(0, result.NumDims);
                    Assert.Equal(expected, (T)result.GetValue());
                }
            }
        }
    }
}
