using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace TensorFlow
{
    public abstract class QueueBase
    {
        public QueueBase(TFSession session)
        {
            Session = session ?? throw new ArgumentNullException(nameof(session));
        }

        protected TFSession Session { get; private set; }
    }

    public class PaddingFIFOQueue : QueueBase
    {
        private TFOutput _handle;
        private TFDataType[] _componentTypes;

        public PaddingFIFOQueue(TFSession session, TFDataType[] componentTypes, TFShape[] shapes, int? capacity = null, string container = null, string operationName = null) 
            : base(session)
        {
            _componentTypes = componentTypes ?? throw new ArgumentNullException(nameof(componentTypes));
            _handle = Session.Graph.PaddingFIFOQueueV2(
                    componentTypes,
                    shapes,
                    capacity,
                    container,
                    operationName);
        }

        public TFOperation Enqueue(TFOutput[] components, long? timeout_ms = null, string operationName = null)
        {
            TFOperation operation = Session.Graph.QueueEnqueueV2(_handle, components, timeout_ms, operationName);
            return operation;
        }

        public TFTensor[] EnqueueExecute(TFOutput[] components, TFTensor[] inputValues, long? timeout_ms = null, string operationName = null)
        {
            TFOperation enqueueOp = Enqueue(components, timeout_ms, operationName);
            TFTensor[] tensors = Session.Run(components, inputValues, Array.Empty<TFOutput>(), new[] { enqueueOp });
            return tensors;
        }

        public TFOutput[] Dequeue(long? timeout_ms = null, string operationName = null)
        {
            TFOutput[] values = Session.Graph.QueueDequeueV2(_handle, _componentTypes, timeout_ms, operationName);
            return values;
        }

        public TFTensor[] DequeueExecute(long? timeout_ms = null, string operationName = null)
        {
            TFOutput[] values = Session.Graph.QueueDequeueV2(_handle, _componentTypes, timeout_ms, operationName);
            TFTensor[] tensors = Session.Run(Array.Empty<TFOutput>(), Array.Empty<TFTensor>(), values);
            return tensors;
        }

        public T[] DequeueExecute<T>(long? timeout_ms = null, string operationName = null)
        {
            return DequeueExecute(timeout_ms, operationName).Select(x => x.GetValue()).Cast<T>().ToArray();
        }

        public TFOutput GetSize(string operationName = null)
        {
            TFOutput sizeOutput = Session.Graph.QueueSizeV2(_handle, operationName);
            return sizeOutput;
        }

        public int GetSizeExecute(string operationName = null)
        {
            TFOutput sizeOutput = GetSize(operationName);
            TFTensor[] tensors = Session.Run(Array.Empty<TFOutput>(), Array.Empty<TFTensor>(), new TFOutput[] { sizeOutput });
            return (int)tensors.First().GetValue();
        }
    }
}