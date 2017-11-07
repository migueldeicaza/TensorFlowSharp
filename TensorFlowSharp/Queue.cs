using System;
using System.Linq;

namespace TensorFlow
{
	/// <summary>
	/// Base class for queue implementations.
	/// Port of Python implementation https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/data_flow_ops.py
	/// </summary>
	public abstract class QueueBase
	{
		/// <summary>
		/// A queue is a TensorFlow data structure that stores tensors across
		/// multiple steps, and exposes operations that enqueue and dequeue
		/// tensors.
		/// Each queue element is a tuple of one or more tensors, where each
		/// tuple component has a static dtype, and may have a static shape.The
		/// queue implementations support versions of enqueue and dequeue that
		/// handle single elements, versions that support enqueuing and
		/// dequeuing a batch of elements at once.
		/// </summary>
		/// <param name="session">Session instance</param>
		public QueueBase (TFSession session)
		{
			Session = session ?? throw new ArgumentNullException (nameof (session));
		}

		/// <summary>
		/// The session that this QueueBased was created for.
		/// </summary>
		/// <value>The session.</value>
		protected TFSession Session { get; private set; }

		/// <summary>
		///   Enqueues a tuple of one or more tensors in this queue.
		/// </summary>
		/// <param name="components">
		///   One or more tensors from which the enqueued tensors should be taken.
		/// </param>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueEnqueueV2'.
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is full, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   Returns the description of the operation
		/// </returns>
		/// <remarks>
		///   The components input has k elements, which correspond to the components of
		///   tuples stored in the given queue.
		/// </remarks>
		public abstract TFOperation Enqueue (TFOutput [] components, long? timeout_ms = null, string operationName = null);

		/// <summary>
		///   Dequeues a tuple of one or more tensors from this queue.
		/// </summary>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is empty, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   One or more tensors that were dequeued as a tuple.
		///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
		/// </returns>
		/// <remarks>
		///   This operation has k outputs, where k is the number of components
		///   in the tuples stored in the given queue, and output i is the ith
		///   component of the dequeued tuple.
		/// </remarks>
		public abstract TFOutput [] Dequeue (long? timeout_ms = null, string operationName = null);

		/// <summary>
		/// Gets the size of this queue.
		/// </summary>
		/// <param name="operationName"></param>
		/// <returns>queue size</returns>
		public abstract TFOutput GetSize (string operationName = null);
	}

	/// <summary>
	/// A FIFOQueue that supports batching variable-sized tensors by padding.
	/// Port of Python implementation https://github.com/tensorflow/tensorflow/blob/b46340f40fe5e2ec9bfcd385b07cfb914055fb51/tensorflow/python/ops/data_flow_ops.py#L697
	/// </summary>
	public class PaddingFIFOQueue : QueueBase
	{
		private TFOutput _handle;
		private TFDataType [] _componentTypes;

		/// <summary>
		/// Creates a queue that dequeues elements in a first-in first-out order.
		/// A `PaddingFIFOQueue` has bounded capacity; supports multiple concurrent
		/// producers and consumers; and provides exactly-once delivery.
		/// A `PaddingFIFOQueue` holds a list of up to `capacity` elements.Each
		/// element is a fixed-length tuple of tensors whose dtypes are
		/// described by `dtypes`, and whose shapes are described by the `shapes`
		/// </summary>
		/// <param name="session"></param>
		/// <param name="componentTypes">The type of each component in a tuple.</param>
		/// <param name="shapes">
		///   Optional argument
		///   The shape of each component in a value. The length of this attr must
		///   be either 0 or the same as the length of component_types.
		///   Shapes of fixed rank but variable size are allowed by setting
		///   any shape dimension to -1.  In this case, the inputs' shape may vary along
		///   the given dimension, and DequeueMany will pad the given dimension with
		///   zeros up to the maximum shape of all elements in the given batch.
		///   If the length of this attr is 0, different queue elements may have
		///   different ranks and shapes, but only one element may be dequeued at a time.</param>
		/// <param name="capacity"> Optional argument. The upper bound on the number of elements in this queue. Negative numbers mean no limit.</param>
		/// <param name="container"> Optional argument. If non-empty, this queue is placed in the given container. Otherwise, a default container is used.</param>
		/// <param name="operationName"> If specified, the created operation in the graph will be this one, otherwise it will be named 'PaddingFIFOQueueV2'.</param>
		public PaddingFIFOQueue (TFSession session, TFDataType [] componentTypes, TFShape [] shapes, int? capacity = null, string container = null, string operationName = null)
			: base (session)
		{
			_componentTypes = componentTypes ?? throw new ArgumentNullException (nameof (componentTypes));
			_handle = Session.Graph.PaddingFIFOQueueV2 (
					componentTypes,
					shapes,
					capacity,
					container,
					operationName);
		}

		/// <summary>
		///   Enqueues a tuple of one or more tensors in this queue.
		/// </summary>
		/// <param name="components">
		///   One or more tensors from which the enqueued tensors should be taken.
		/// </param>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueEnqueueV2'.
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is full, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   Returns the description of the operation
		/// </returns>
		/// <remarks>
		///   The components input has k elements, which correspond to the components of
		///   tuples stored in the given queue.
		///   
		///   N.B. If the queue is full, this operation will block until the given
		///   element has been enqueued (or 'timeout_ms' elapses, if specified).
		/// </remarks>
		public override TFOperation Enqueue (TFOutput [] components, long? timeout_ms = null, string operationName = null)
		{
			TFOperation operation = Session.Graph.QueueEnqueueV2 (_handle, components, timeout_ms, operationName);
			return operation;
		}

		/// <summary>
		///   Enqueues a tuple of one or more tensors in this queue and runs the session.
		/// </summary>
		/// <param name="components">
		///   One or more tensors from which the enqueued tensors should be taken.
		/// </param>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueEnqueueV2'.
		/// </param>
		/// <param name="inputValues">
		///   Values to enqueue
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is full, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   Returns the description of the operation
		/// </returns>
		/// <remarks>
		///   The components input has k elements, which correspond to the components of
		///   tuples stored in the given queue.
		/// </remarks>
		public TFTensor [] EnqueueExecute (TFOutput [] components, TFTensor [] inputValues, long? timeout_ms = null, string operationName = null)
		{
			TFOperation enqueueOp = Enqueue (components, timeout_ms, operationName);
			TFTensor [] tensors = Session.Run (components, inputValues, Array.Empty<TFOutput> (), new [] { enqueueOp });
			return tensors;
		}

		/// <summary>
		///   Dequeues a tuple of one or more tensors from the given queue.
		/// </summary>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is empty, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   One or more tensors that were dequeued as a tuple.
		///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
		/// </returns>
		/// <remarks>
		///   This operation has k outputs, where k is the number of components
		///   in the tuples stored in the given queue, and output i is the ith
		///   component of the dequeued tuple.
		/// </remarks>
		public override TFOutput [] Dequeue (long? timeout_ms = null, string operationName = null)
		{
			TFOutput [] values = Session.Graph.QueueDequeueV2 (_handle, _componentTypes, timeout_ms, operationName);
			return values;
		}

		/// <summary>
		///   Dequeues a tuple of one or more tensors from this queue and runs the session.
		/// </summary>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is empty, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   One or more tensors that were dequeued as a tuple.
		///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
		/// </returns>
		/// <remarks>
		///   This operation has k outputs, where k is the number of components
		///   in the tuples stored in the given queue, and output i is the ith
		///   component of the dequeued tuple.
		/// </remarks>
		public TFTensor [] DequeueExecute (long? timeout_ms = null, string operationName = null)
		{
			TFOutput [] values = Session.Graph.QueueDequeueV2 (_handle, _componentTypes, timeout_ms, operationName);
			TFTensor [] tensors = Session.Run (Array.Empty<TFOutput> (), Array.Empty<TFTensor> (), values);
			return tensors;
		}

		/// <summary>
		///   Dequeues elements from this queue and cast all elements to specific T type. It can be use when all elements in the queue of the same T type
		/// </summary>
		/// <param name="operationName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueDequeueV2'.
		/// </param>
		/// <param name="timeout_ms">
		///   Optional argument
		///   If the queue is empty, this operation will block for up to
		///   timeout_ms milliseconds.
		///   Note: This option is not supported yet.
		/// </param>
		/// <returns>
		///   
		/// </returns>
		public T [] DequeueExecute<T> (long? timeout_ms = null, string operationName = null)
		{
			return DequeueExecute (timeout_ms, operationName).Select (x => x.GetValue ()).Cast<T> ().ToArray ();
		}

		/// <summary>
		/// Gets the size of this queue.
		/// </summary>
		/// <param name="operationName">If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueSizeV2'.</param>
		/// <returns>queue size</returns>
		public override TFOutput GetSize (string operationName = null)
		{
			TFOutput sizeOutput = Session.Graph.QueueSizeV2 (_handle, operationName);
			return sizeOutput;
		}

		/// <summary>
		/// Uses provided session instance to obtain the size of this queue
		/// </summary>
		/// <param name="operationName">If specified, the created operation in the graph will be this one, otherwise it will be named 'QueueSizeV2'.</param>
		/// <returns>number of elements in the queue</returns>
		public int GetSizeExecute (string operationName = null)
		{
			TFOutput sizeOutput = GetSize (operationName);
			TFTensor [] tensors = Session.Run (Array.Empty<TFOutput> (), Array.Empty<TFTensor> (), new TFOutput [] { sizeOutput });
			return (int)tensors.First ().GetValue ();
		}
	}
}