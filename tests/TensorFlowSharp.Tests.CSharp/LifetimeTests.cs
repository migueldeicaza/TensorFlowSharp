using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class LifetimeTests
    {

        public class TFOperationLifetime
        {
            [Fact]
            public void DisposingGraphMakesOperationInvalid()
            {
                TFOperation op;
                using (var graph = new TFGraph())
                {
                    op = graph.Const(10).Operation;
                }

                var e = Record.Exception(() => op.ControlOutputs);

                Assert.IsType<ObjectDisposedException>(e);
            }

            private static object[] BuildOpTestcase(Action<TFOperation> op) => new[] { op };
            private static IEnumerable<object[]> OperationsThatShouldThrow()
            {
                yield return BuildOpTestcase(op => { var foo = op.ControlOutputs; });
                yield return BuildOpTestcase(op => { var foo = op.GetAttributeMetadata(""); });
                yield return BuildOpTestcase(op => { var foo = op.GetAttributeShape("", 1); });
                yield return BuildOpTestcase(op => { var foo = op.GetAttributeType(""); });
                yield return BuildOpTestcase(op => { var foo = op.GetInput(0); });
            }


            [Theory]
            [MemberData(nameof(OperationsThatShouldThrow))]
            public void AfterDisposingGraphCertainOperationsFail(Action<TFOperation> operation)
            {
                TFOperation op;
                using (var graph = new TFGraph())
                {
                    op = graph.Const(10).Operation;
                }

                var e = Record.Exception(() => operation(op));

                Assert.IsType<ObjectDisposedException>(e);
            }



            private static IEnumerable<object[]> OperationsThatShouldNotThrow()
            {
                yield return BuildOpTestcase(op => { var foo = op.Name; });
                yield return BuildOpTestcase(op => { var foo = op.Handle; });
                yield return BuildOpTestcase(op => { var foo = op.NumControlInputs; });
                yield return BuildOpTestcase(op => { var foo = op.NumControlOutputs; });
                yield return BuildOpTestcase(op => { var foo = op.NumInputs; });
                yield return BuildOpTestcase(op => { var foo = op.NumOutputs; });
            }

            [Theory]
            [MemberData(nameof(OperationsThatShouldNotThrow))]
            public void AfterDisposingGraphCertainOperationsStillSucceeds(Action<TFOperation> operation)
            {
                TFOperation op;
                using (var graph = new TFGraph())
                {
                    op = graph.Const(10).Operation;
                }

                var record = Record.Exception(() => operation(op));

                Assert.Null(record);
            }
        }

        public class TFOutputLifetime
        {
            [Fact]
            public void AfterDisposingGraphDatatypeBecomesUnknown()
            {
                var sut = CreateOutputAndDisposeGraph();

                var datatype = sut.OutputType;

                Assert.Equal(TFDataType.Unknown, datatype);
            }

            [Fact]
            public void AfterDisposingGraphCertainOperationsAreNotValidOnOutput()
            {
                var sut = CreateOutputAndDisposeGraph();

                var ex1 = Record.Exception(() => sut.NumConsumers);
                var ex2 = Record.Exception(() => sut.Operation);
                var ex3 = Record.Exception(() => sut.OutputConsumers);

                Assert.IsType<ObjectDisposedException>(ex1);
                Assert.IsType<ObjectDisposedException>(ex2);
                Assert.IsType<ObjectDisposedException>(ex3);
            }

            private static TFOutput CreateOutputAndDisposeGraph()
            {
                using(var graph = new TFGraph())
                {
                    return graph.Const(10);
                }
            }
        }
    }
}
