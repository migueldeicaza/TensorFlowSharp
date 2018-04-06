//
// Port of the optimizer framework from Python
//
// Authors:
//   Miguel de Icaza
//

using System;
namespace TensorFlow {
	public partial class TFGraph {
		/// <summary>
		/// Sums `values` associated with any non-unique `indices`
		/// </summary>
		/// <param name="values">Tensor with rank bigger than 1</param>
		/// <param name="indices">A one-dimensional integer tensor indexing into the first dimension of values (as in an IndexedSliced object).</param>
		(TFOutput summedValues, TFOutput uniqueIndices) DeduplicateIndexedSlices (TFOutput values, TFOutput indices)
		{
			(var unique_indices, var new_index_positions) = Unique (indices);
			var shape = Shape (unique_indices);

			// Extract the first dimension, like a slice shape[0] in python, I need
			// to implement that on TFOutput as a helper indexer.
			var shape1 = StridedSlice (
				shape,
				begin:   Const (new int [] { 0 }),
				end:     Const (new int [] { 0 }),
				strides: Const (new int [] { 1 }),
				begin_mask: 1,
				end_mask: 1);

			var summed_values = UnsortedSegmentSum (values, new_index_positions, shape1);
			return (summed_values, unique_indices);
		}
	}

	// Interface for abstracting over variables in the optimizers.
	class OptimizableVariable
	{
		protected Variable Var;

		public OptimizableVariable (Variable var)
		{
			Var = var;
		}

		// Returns the optimization target for this variable.
		protected virtual TFOutput Target => Var.VariableOp;

		// Returns the update ops for updating the variable.
		protected virtual object UpdateOp (Optimizer optimizer, TFGraph graph)
		{
			throw new NotImplementedException ();
		}
	}

	// Processor for variable
	class RefVariableProcessor : OptimizableVariable
	{
		public RefVariableProcessor (Variable var) : base (var) { }
		public override string ToString () => $"<RefVariableProcessor: {Var}>";
	}

	// Processor for dense ResourceVariables.
	class DenseReadResourceVariableProcessor : OptimizableVariable
	{
		public DenseReadResourceVariableProcessor (Variable var) : base (var) { }
	}

	// Processor for dense ResourceVariables.
	class DenseResourceVariableProcessor : OptimizableVariable
	{
		public DenseResourceVariableProcessor (Variable var) : base (var) { }
	}

	class StreamingModelPortProcessor : OptimizableVariable
	{
		public StreamingModelPortProcessor (Variable var) : base (var) { }
	}

	//"""Processor for ordinary Tensors.
	//
	// Even though a Tensor can't really be updated, sometimes it is useful to
  	// compute the gradients with respect to a Tensor using the optimizer.Updating
  	// the Tensor is, of course, unsupported.
	class TensorProcessor : OptimizableVariable
	{
		TensorProcessor (Variable var) : base (var) { }
		protected override object UpdateOp (Optimizer optimizer, TFGraph graph)
		{
			throw new NotImplementedException ("Not supported on tensors");
		}
	}

	public class Optimizer {

		// What should the parameter be?   The Python code calls
		// the equivalent of x.Operation.Optype, which means that it would be
		// a TFOutput, but the classes above are designed to "dereference" the 
		// value, so I suspect they expect a higher-level operation, like
		// our own Variable.
		OptimizableVariable GetProcessor ()
		{
			//if (var.Operation.OpType == "VarHandleOp"){
			//	return new DenseResourceVariableProcessor (
			// if VarHandleOp -> _DenseResourceVariableProcessor
			// if x is variables.Variable -> _RefVariableProcessor
			// if SubmodelPort -> _StreamingModelPortProcessor
			// if Tensor -> _TensorProcessor
			return null;
		}

		public Optimizer ()
		{
			
		}
	}
}
