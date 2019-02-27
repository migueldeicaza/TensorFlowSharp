using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{

    public class TFOperationTests
    {
        [Fact]
        public void InputsEqualToTheIncommingNumberOfOutputsAreRetrievedInOrder()
        {
            using (var graph = new TFGraph())
            {
                var a = graph.Const(0f);
                var v2 = graph.Variable(graph.Const(0.6f));

                var add = graph.Add(a, v2.Read);

                AssertExpectedNumberOfInputs(0, a.Operation);
                AssertExpectedNumberOfInputs(1, v2.Read.Operation);
                AssertExpectedNumberOfInputs(2, add.Operation);
            }
        }


        private static void AssertExpectedNumberOfInputs(int expected, TFOperation sut)
        {
            Assert.Equal(expected, sut.NumInputs);

            var inputs = sut.Inputs;

            Assert.NotNull(inputs);
            Assert.Equal(expected, inputs.Length);

            for(int i = 0; i < inputs.Length; ++i)
            {
                var input = inputs[i];
                Assert.Equal(i, input.Index);
                Assert.Equal(sut.Handle, input.Operation);
            }
        }
    }
}
