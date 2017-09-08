using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class PartialRunTests
    {
        [Fact]
        public void Should_RunPartialRun()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                float aValue = 1;
                float bValue = 2;

                var a = graph.Placeholder(TFDataType.Float);
                var b = graph.Placeholder(TFDataType.Float);
                var c = graph.Placeholder(TFDataType.Float);

                var r1 = graph.Add(a, b);
                var r2 = graph.Mul(r1, c);

                var h = session.PartialRunSetup(new[] { a, b, c }, new[] { r1, r2 }, new[] { r1.Operation, r2.Operation });
                var res = session.PartialRun(h, new[] { a, b }, new TFTensor[] { aValue, bValue }, new TFOutput[] { r1 }, new[] { r1.Operation }); // 1+2=3
                var calculated = (float)res[0].GetValue();
                Assert.Equal(3, calculated);

                float temp = calculated * 17; // 3*17=51
                res = session.PartialRun(h, new[] { c }, new TFTensor[] { temp }, new[] { r2 }, new[] { r2.Operation }); // 51*3=153
                calculated = (float)res[0].GetValue();
                Assert.Equal(153, calculated);
            }
        }
    }
}
