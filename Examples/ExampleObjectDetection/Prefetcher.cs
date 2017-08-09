using System.Collections.Generic;
using System.Linq;
using TensorFlow;

namespace ExampleObjectDetection
{
    public static class Prefetcher
    {
        public static void Prefetch(TFGraph graph, Dictionary<string, string> tensorDictionary, int capacity)
        {
            //var names = tensorDictionary.Keys;
            //TFDataType[] dtypes = tensorDictionary.Values.Select(x => x.dtype).ToArray();
            //TFShape[] shapes = tensorDictionary.Values.Select(x => x.GetShape()).ToArray();

            //var prefetchQueue = graph.PaddingFIFOQueueV2(dtypes, shapes, capacity, operName: "prefetch_queue");

            //graph.Sca
        }
    }
}