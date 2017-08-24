using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class BitwiseOperationTests
    {
        [Theory]
        [InlineData(2, 3, 2)]
        [InlineData(3, 0, 0)]
        [InlineData(1, 3, 1)]
        public void Should_EvaluateBitwiseAnd(int aValue, int bValue, int expected)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var a = graph.Placeholder(TFDataType.Int32);
                var b = graph.Placeholder(TFDataType.Int32);

                TFOutput y = graph.BitwiseAnd(a, b);

                TFTensor[] result = session.Run(new[] {a, b}, new TFTensor[] {aValue, bValue}, new[] {y});

                Assert.Equal(expected, (int) result[0].GetValue());
            }
        }
    }
}
