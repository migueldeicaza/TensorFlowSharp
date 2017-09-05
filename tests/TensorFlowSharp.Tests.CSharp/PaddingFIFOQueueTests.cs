using System.Collections.Generic;
using System.Linq;
using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class PaddingFIFOQueueTests
    {
        [Fact]
        public void Should_EnqueueAndDequeue_ScalarValues()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                int[] numbersToEnqueue = new int[] { 5, 8, 9 };

                TFOutput a = graph.Placeholder(TFDataType.Int32);
                TFOutput b = graph.Placeholder(TFDataType.Int32);
                TFOutput c = graph.Placeholder(TFDataType.Int32);
                var queue = new PaddingFIFOQueue(session, new[] { TFDataType.Int32 }, new[] { TFShape.Scalar });
                queue.EnqueueExecute(new[] { a }, new[] { (TFTensor)numbersToEnqueue[0] });
                queue.EnqueueExecute(new[] { b }, new[] { (TFTensor)numbersToEnqueue[1] });
                queue.EnqueueExecute(new[] { c }, new[] { (TFTensor)numbersToEnqueue[2] });
                int size = queue.GetSizeExecute();
                Assert.Equal(numbersToEnqueue.Length, size);

                List<int> dequeuedNumbers = new List<int>();
                dequeuedNumbers.Add(queue.DequeueExecute<int>().Single());
                dequeuedNumbers.Add(queue.DequeueExecute<int>().Single());
                dequeuedNumbers.Add(queue.DequeueExecute<int>().Single());

                Assert.Equal(numbersToEnqueue, dequeuedNumbers.ToArray());
            }
        }
    }
}