using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorFlow
{
	public partial class TFGraph
	{
		/// <summary>
		/// Creates a constant operation from a TFTensor or constant
		/// </summary>
		/// <param name="value">Value.</param>
		/// <param name="operName">Oper name.</param>
		/// <remarks>
		/// Since TFTensor have implicit conversion operators, you can call this method with
		/// a constant like this: graph.Const (23)
		/// </remarks>
		public TFOutput Const (TFTensor value, string operName = null)
		{
			return Const (value, value.TensorType, operName);
		}

		// Returns range(0, rank(x)) if reduction_indices is null
		TFOutput ReduceDims (TFOutput input, TFOutput? axis = null)
		{
			if (axis.HasValue)
				return axis.Value;

			// Fast path: avoid creating Rank and Range ops if ndims is known.
			var shape = GetTensorShape (input);
			if (shape.IsFullySpecified) {
				// The python code distinguishes between tensor and sparsetensor

				var array = new int [shape.NumDimensions];
				for (int i = 0; i < array.Length; i++)
					array [i] = i;

				return this.Const (array, TFDataType.Int32);
			}
			// Otherwise, we rely on Range and Rank to do the right thing at run-time.
			return Range (Const (0), Rank (input), Const (1));
		}

		/// <summary>
		/// Computes the sum of elements across dimensions of a tensor.
		/// </summary>
		/// <returns>The reduced tensor.</returns>
		/// <param name="input">The tensor to reduce. Should have numeric type.</param>
		/// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
		/// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
		/// <param name="operName">A name for the operation, optional.</param>
		/// <remarks>
		///   Reduces input_tensor along the dimensions given in axis.
		/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
		/// entry in axis. If keep_dims is true, the reduced dimensions
		/// are retained with length 1.
		/// 
		/// If axis has no entries, all dimensions are reduced, and a
		/// tensor with a single element is returned.
		/// </remarks>
		public TFOutput ReduceSum (TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
		{
			return Sum (input, this.ReduceDims (input, axis), keep_dims, operName);
		}

		/// <summary>
		/// Computes the product of elements across dimensions of a tensor.
		/// </summary>
		/// <returns>The reduced tensor.</returns>
		/// <param name="input">The tensor to reduce. Should have numeric type.</param>
		/// <param name="axis">The dimensions to reduce. If not se (the default), reduces all dimensions.</param>
		/// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
		/// <param name="operName">A name for the operation, optional.</param>
		/// <remarks>
		///   Reduces input_tensor along the dimensions given in axis.
		/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
		/// entry in axis. If keep_dims is true, the reduced dimensions
		/// are retained with length 1.
		/// 
		/// If axis has no entries, all dimensions are reduced, and a
		/// tensor with a single element is returned.
		/// </remarks>
		public TFOutput ReduceProd (TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
		{
			return Prod (input, this.ReduceDims (input, axis), keep_dims, operName);
		}

		/// <summary>
		/// Computes the mean of elements across dimensions of a tensor.
		/// </summary>
		/// <returns>The reduced tensor.</returns>
		/// <param name="input">The tensor to reduce. Should have numeric type.</param>
		/// <param name="axis">The dimensions to reduce. If not set (the default), reduces all dimensions.</param>
		/// <param name="keep_dims">If set to <c>true</c> retains reduced dimensions with length 1.</param>
		/// <param name="operName">A name for the operation, optional.</param>
		/// <remarks>
		/// <para>
		///   Reduces input_tensor along the dimensions given in axis.
		/// Unless keep_dims is true, the rank of the tensor is reduced by 1 for each
		/// entry in axis. If keep_dims is true, the reduced dimensions
		/// are retained with length 1.</para>
		/// 
		/// <para>
		/// If axis has no entries, all dimensions are reduced, and a
		/// tensor with a single element is returned.</para>
		/// </remarks>
		public TFOutput ReduceMean (TFOutput input, TFOutput? axis = null, bool? keep_dims = false, string operName = null)
		{
			if (input.OutputType == TFDataType.Bool)
				input = this.Cast (input, TFDataType.Int8);
			return this.Mean (input, this.ReduceDims (input, axis), keep_dims, operName);
		}


		// Helper method to create a variable and track it.
		Variable MakeVariable (TFOutput initialValue, bool trainable, string operName)
		{
			var scopeName = MakeName ("Variable", operName);

			using (var newScope = WithScope (scopeName)) {
				var type = initialValue.OutputType;
				var variableHandle = VarHandleOp (type, new TFShape (GetShape (initialValue)), shared_name: operName);
				using (var aScope = WithScope ("Assign")) {
					var assignOp = AssignVariableOp (variableHandle, initialValue);
					using (var rScope = WithScope ("Read")) {
						var readHandle = ReadVariableOp (variableHandle, type);

						var nv = new Variable (variableHandle, readHandle, assignOp);
						if (trainable)
							AddTrainableVariable (nv);
						AddInitVariable (assignOp);
						return nv;
					}
				}
			}

		}

		/// <summary>
		/// Variable node, with a starting initial value.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="init">Returns the operation that initializes the value of the variable.</param>
		/// <param name="value">Returns the value of the variable.</param>
		/// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable
		/// </remarks>
		public Variable Variable (TFOutput initialValue, out TFOperation init, out TFOutput value, bool trainable = true, string operName = null)
		{
			var nv = MakeVariable (initialValue, trainable, operName);
			init = nv.Assign;
			value = nv.Read;
			return nv;
		}

		List<TFOperation> pending_init_variables;
		List<Variable> trainable_variables;

		/// <summary>
		/// Registers a specified variable as an initialization variable.
		/// </summary>
		/// <param name="variable">Variable to register.</param>
		/// <remarks>
		/// <para>
		/// This is a convenience method to track the variables that need to be initialized in the graph,
		/// you can retrieve the list of all those variables by calling the <see cref="M:TensorFlow.TFGraph.GetGlobalVariablesInitializer"/>
		/// which will return this list and clear the state at that point.
		/// </para>
		/// <para>
		/// You typically use this method from helper methods to register all the variables that you want
		/// initialized, and a higher level method will retrieve all these variables and initialize them
		/// at their convenience.
		/// </para>
		/// </remarks>
		public void AddInitVariable (TFOperation variable)
		{
			if (pending_init_variables == null)
				pending_init_variables = new List<TFOperation> ();
			pending_init_variables.Add (variable);
		}

		// TODO: finalize semantics, when should we clear these?
		internal void AddTrainableVariable (Variable variable)
		{
			if (trainable_variables == null)
				trainable_variables = new List<Variable> ();
			trainable_variables.Add (variable);
		}

		/// <summary>
		/// Gets the list of all registered global variables.
		/// </summary>
		/// <returns>The array of variables that should be initialized.</returns>
		/// <remarks>
		/// After this method is invoked the list of pending initialization variables
		/// is cleared.
		/// </remarks>
		public TFOperation [] GetGlobalVariablesInitializer ()
		{
			if (pending_init_variables == null)
				pending_init_variables = new List<TFOperation> ();
			var res = pending_init_variables.ToArray ();
			pending_init_variables.Clear ();
			return res;
		}

        /// <summary>
        /// Gets the list of all registered trainable variables.
		/// </summary>
		/// <returns>The array of variables that should be trained.</returns>
		public Variable[] GetTrainableVariables() => trainable_variables.ToArray();

        /// <summary>
        /// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
        /// </summary>
        /// <param name="initialValue">Initial value.</param>
        /// <param name="value">Returns the value of the variable.</param>
        /// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
        /// <param name="operName">Operation name, optional.</param>
        /// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
        /// <remarks>
        /// Variables need to be initialized before the main execution so you will typically want to
        /// run the session on the variable.
        /// 
        /// The init sequence for the variable is stored in the graph, you must manually initialize 
        /// those by running the session on the global variables.
        /// </remarks>
        public Variable Variable (TFOutput initialValue, out TFOutput value, bool trainable = true, string operName = null)
		{
			var nv = MakeVariable (initialValue, trainable, operName);
			value = nv.Read;
			return nv;
		}

		/// <summary>
		/// Variable node, with a starting initial value.  Convenience that registers the init variable to a global queue.
		/// </summary>
		/// <param name="initialValue">Initial value.</param>
		/// <param name="trainable">If true, this add the variable to the graph's TrainableVariables, this collection is intended to be used by the Optimizer classes.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>The returning Variable contains the variable, with three nodes with the operations making up the variable assignment.</returns>
		/// <remarks>
		/// Variables need to be initialized before the main execution so you will typically want to
		/// run the session on the variable.
		/// 
		/// The init sequence for the variable is stored in the graph, you must manually initialize 
		/// those by running the session on the global variables.
		/// </remarks>
		public Variable Variable (TFOutput initialValue, bool trainable = true, string operName = null)
		{
			return MakeVariable (initialValue, trainable, operName);
		}

		//
		// Converts a shape to a tensor, to a TFOutput
		//
		TFOutput ShapeTensorOutput (TFShape shape)
		{
			if (shape.IsLongArray)
				return Const (shape.ToArray (), TFDataType.Int64);
			else
				return Const (shape.ToIntArray (), TFDataType.Int32);
		}

		/// <summary>
		/// Gets or sets the graph random seed, see remarks for details.
		/// </summary>
		/// <value>The seed.</value>
		/// <remarks>
		///  Operations that rely on a random seed actually derive it from two seeds:
		///  the graph-level and operation-level seeds.This sets the graph-level seed.
		///
		/// Its interactions with operation-level seeds is as follows:
		/// 1. If neither the graph-level nor the operation seed is set:
		///    A random seed is used for this op.
		/// 2. If the graph-level seed is set, but the operation seed is not:
		///    The system deterministically picks an operation seed in conjunction
		///    with the graph-level seed so that it gets a unique random sequence.
		/// 3. If the graph-level seed is not set, but the operation seed is set:
		///    A default graph-level seed and the specified operation seed are used to
		///    determine the random sequence.
		/// 4. If both the graph-level and the operation seed are set:
		///    Both seeds are used in conjunction to determine the random sequence.
		/// </remarks>
		public int? Seed { get; set; }

		/// <summary>
		/// Returns the graph and local seeds based on an optionally set incoming seed value.
		/// </summary>
		/// <param name="operationSeed">The seed value that might be set.</param>
		/// <param name="graphSeed">Returned graph seed.</param>
		/// <param name="localSeed">Returned local seed.</param>
		/// <remarks>
		/// This helper function returns two seeds derived from graph-level and op-level seeds.
		/// Many random operations internally use the two seeds to allow user to change 
		/// the seed globally for a graph, or for only specific operations.
		/// </remarks>
		public void GetRandomSeeds (int? operationSeed, out int graphSeed, out int localSeed)
		{
			if (Seed.HasValue) {
				graphSeed = Seed.Value;
				if (operationSeed.HasValue)
					localSeed = operationSeed.Value;
				else
					localSeed = LastId;
			} else {
				graphSeed = 87654321;
				if (operationSeed.HasValue) {
					localSeed = operationSeed.Value;
				} else {
					localSeed = 0;
				}
			}
		}

		/// <summary>
		/// Computes dropout. 
		/// </summary>
		/// <param name="x">A tensor.</param>
		/// <param name="keep_prob">A scalar Tensor with the same type as x. The probability that each element is kept.</param>
		/// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
		/// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <remarks>
		/// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
		/// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
		/// </remarks>
		public TFOutput Dropout (TFOutput x, TFOutput keep_prob, TFShape noise_shape = null, int? seed = null, string operName = null)
		{
			var scopeName = MakeName ("dropout", operName);

			using (var newScope = WithScope (scopeName)) {
				if (noise_shape == null)
					noise_shape = new TFShape (GetShape (x));

				TFOutput shapeTensor = ShapeTensorOutput (noise_shape);

				// uniform [keep_prob, 1.0 + keep_prob)
				TFOutput random_tensor = keep_prob;
				random_tensor = Add (random_tensor, RandomUniform (shapeTensor, seed: seed, dtype: x.OutputType));

				// 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
				TFOutput binary_tensor = Floor (random_tensor);
				TFOutput ret = Mul (Div (x, keep_prob), binary_tensor);
				SetTensorShape (ret, GetShape (x));
				return ret;
			}
		}

		/// <summary>
		/// Computes dropout. 
		/// </summary>
		/// <param name="x">A tensor.</param>
		/// <param name="keep_prob">A scalar Tensor with the same type as x. The probability that each element is kept.</param>
		/// <param name="noise_shape">A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.</param>
		/// <param name="seed">Integer seed used for the random distribution, using the TensorFlow SetRandomSeed .</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <remarks>
		/// With probability keep_prob, outputs the input element scaled up by 1 / keep_prob, 
		/// otherwise outputs 0. The scaling is so that the expected sum is unchanged.
		/// </remarks>
		public TFOutput Dropout (TFOutput x, double keep_prob, TFShape noise_shape = null, int? seed = null, string operName = null)
		{
			if (keep_prob < 0 || keep_prob >= 1)
				throw new ArgumentOutOfRangeException ("keep_prob must be a scalar tensor or a float in the range (0, 1], got " + keep_prob);

			if (keep_prob == 1)
				return x;

			var scopeName = MakeName ("dropout", operName);
			using (var newScope = WithScope (scopeName)) {
				var tkeep_prob = Const (keep_prob);
				return Dropout (x, tkeep_prob, noise_shape, seed, operName);
			}
		}


		/// <summary>
		/// Clips tensor values to a maximum L2-norm.
		/// </summary>
		/// <remarks>
		/// <para>
		/// Given a tensor <paramref name="x"/>, and a maximum clip value <paramref name="clip_norm"/>, this operation normalizes 
		/// <paramref name="x"/> so that its L2-norm is less than or equal to <paramref name="clip_norm"/>, along the dimensions 
		/// given in <paramref name="axes"/>. Specifically, in the default case where all dimensions are used for calculation, if
		/// the L2-norm of <paramref name="x"/> is already less than or equal to <paramref name="clip_norm"/>, then <paramref name="x"/>
		/// is not modified. If the L2-norm is greater than <paramref name="clip_norm"/>, then this operation returns a tensor of 
		/// the same type and shape as <paramref name="x"/> with its values set to: <c>t* clip_norm / l2norm(t)</c></para>
		/// </remarks>
		/// <param name="x">The tensor.</param>
		/// <param name="clip_norm">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
		/// <param name="axes">The minimum value to clip by. A 0 - D(scalar) tensor, or a tensor with the same shape as <paramref name="x"/>.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>A clipped <see cref="TFOutput">tensor</see>.</returns>
		public TFOutput ClipByNorm (TFOutput x, TFOutput clip_norm, TFOutput? axes = null, string operName = null)
		{
			// https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L73
			var scopeName = MakeName ("ClipByNorm", operName);
			using (var newScope = WithScope (scopeName)) {
				// Calculate L2-norm, clip elements by ratio of clip_norm to L2-norm
				var l2norm_inv = Rsqrt (ReduceSum (Mul (x, x), axes, keep_dims: true));
				var intermediate = Mul (x, clip_norm);

				var tclip = Identity (Mul (intermediate, Minimum (l2norm_inv, Div (Const (new TFTensor (1.0)), clip_norm), operName: operName)));

				return tclip;
			}
		}

		/// <summary>
		/// Computes the global norm of multiple tensors.
		/// </summary>
		/// <remarks>
		/// <para>
		///  Given a tuple or list of tensors <paramref name="tensors"/>, this operation returns the global norm of the elements in all tensors 
		///  in <paramref name="tensors"/>. The global norm is computed as: <c>global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))</c>. Any 
		///  entries in <paramref name="tensors"/> that are of type None are ignored.</para>
		/// </remarks>
		/// <param name="tensors">The input tensors.</param>
		/// <param name="operName">Operation name, optional.</param>
		/// <returns>A clipped <see cref="TFOutput">tensor</see>.</returns>
		public TFOutput GlobalNorm (TFOutput [] tensors, string operName = null)
		{
			// https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L122
			var scopeName = MakeName ("GlobalNorm", operName);
			using (var newScope = WithScope (scopeName)) {
				TFOutput [] half_squared_norms = new TFOutput [tensors.Length];

				for (int i = 0; i < half_squared_norms.Length; i++)
					half_squared_norms [i] = L2Loss (tensors [i]);

				TFOutput half_squared_norm = ReduceSum (Stack (half_squared_norms));
				TFOutput norm = Sqrt (Mul (half_squared_norm, Const (2.0)), operName: "global_norm");
				return norm;
			}
		}

		/// <summary>
		/// Clips tensor values to a maximum average L2-norm.
		/// </summary>
		/// <remarks>
		/// Given a tensor <paramref name="x"/>, and a maximum clip value <paramref name="clip_norm"/>, this operation 
		/// normalizes <paramref name="x"/> so that its its average L2-norm is less than or equal to <paramref name="clip_norm"/>.
		/// Specifically, if the average L2-norm is already less than or equal to <paramref name="clip_norm"/>, then <paramref name="x"/>
		/// is not modified. If the average L2-norm is greater than <paramref name="clip_norm"/>, then this operation returns a tensor of the same
		/// type and shape as <paramref name="x"/> with its values set to: <c>t* clip_norm / l2norm_avg(t)</c>. In this case, 
		/// the average L2-norm of the output tensor is <paramref name="clip_norm"/>.
		/// </remarks>
		/// <param name="x">The input tensor.</param>
		/// <param name="clip_norm">A maximum clipping value.</param>
		/// <param name="operName">Name of the oper.</param>
		public TFOutput ClipByAverageNorm (TFOutput x, TFOutput clip_norm, string operName = null)
		{
			// https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/clip_ops.py#L251
			var scopeName = MakeName ("ClipByAverageNorm", operName);
			using (var newScope = WithScope (scopeName)) {
				// Calculate L2-norm per element, clip elements by ratio of clip_norm to
				// L2-norm per element
				TFOutput n_element = Cast (Size (x), TFDataType.Float);
				TFOutput l2norm_inv = Rsqrt (ReduceSum (Mul (x, x), Range (Rank (x))));
				TFOutput tclip = Identity (Mul (Mul (x, clip_norm), Minimum (Mul (l2norm_inv, n_element), Div (Const (new TFTensor (1.0)), clip_norm)), operName: operName));

				return tclip;
			}
		}

		/// <summary>
		///   Computes sigmoid cross entropy given `logits`.
		/// </summary>
		/// 
		/// <remarks>
		///    Measures the probability error in discrete classification tasks in which each
		///    class is independent and not mutually exclusive.For instance, one could
		///    perform multilabel classification where a picture can contain both an elephant
		///    and a dog at the same time.
		/// </remarks>
		/// 
		public TFOutput SigmoidCrossEntropyWithLogits (TFOutput labels, TFOutput logits, string operName = null)
		{
			// https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/nn_impl.py#L100

			var scopeName = this.MakeName ("logistic_loss", operName);
			using (var newScope = this.WithScope (scopeName)) {
				// Note: The following lines have not been ported from the original TF implementation since 
				// TensorFlowSharp API should guarantee that logits and labels are of type TFOutput by design:
				//
				//   logits = ops.convert_to_tensor(logits, name: "logits");
				//   labels = ops.convert_to_tensor(labels, name: "labels");
				//   try
				//   {
				//       labels.get_shape().merge_with(logits.get_shape())
				//   }
				//   catch
				//   {
				//       throw new ArgumentException("logits and labels must have the same shape ({logits.get_shape()} vs {labels.get_shape()})");
				//   }

				// The logistic loss formula from above is
				// x - x * z + log(1 + exp(-x))
				// For x < 0, a more numerically stable formula is
				//   -x * z + log(1 + exp(x))
				// Note that these two expressions can be combined into the following:
				// max(x, 0) - x * z + log(1 + exp(-abs(x)))
				// To allow computing gradients at zero, we define custom versions of max and
				// abs functions.
				TFOutput zeros = this.ZerosLike (logits);
				TFOutput cond = this.GreaterEqual (logits, zeros);
				TFOutput relu_logits = this.Where (cond, logits, zeros);
				TFOutput neg_abs_logits = this.Where (cond, this.Neg (logits), logits);
				return this.Add (
					this.Sub (relu_logits, this.Mul (logits, labels)),
					this.Log1p (this.Exp (neg_abs_logits)),
					operName: operName);
			}
		}

		/// <summary>
		///   Shuffle dimensions of x according to a permutation.
		/// </summary>
		/// <param name="x">
		/// </param>
		/// <param name="operName">
		///   If specified, the created operation in the graph will be this one, otherwise it will be named 'Transpose'.
		/// </param>
		/// <returns>
		///   The TFOperation can be fetched from the resulting TFOutput, by fethching the Operation property from the result.
		/// </returns>
		/// <remarks>
		///   The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
		///     `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
		/// </remarks>
		public TFOutput Transpose (TFOutput x, string operName = null)
		{
			TFOutput rank = Rank (x);
			TFOutput perm = Sub (Sub (rank, Const (1)), Range (Const (0), rank, Const (1)));

			return Transpose (x: x, perm: perm, operName: operName);
		}

		/// <summary>
		///   Returns <paramref name="true_fn"/> if the predicate <paramref name="pred"/> is <c>true</c> else <paramref name="false_fn"/>.
		/// </summary>
		/// <param name="pred">A scalar determining whether to return the result of true_fn or false_fn.</param>
		/// <param name="true_fn">The callable to be performed if pred is true.</param>
		/// <param name="false_fn">The callable to be performed if pred is false.</param>
		/// <param name="operName">Optional name prefix for the returned tensors.</param>
		/// <returns>TFOutput.</returns>
		public TFOutput Cond (TFOutput pred, Func<TFOutput> true_fn, Func<TFOutput> false_fn, string operName = null)
		{
			using (WithScope (this.MakeName ("cond", operName))) {
				// Add the Switch to the graph.
				(TFOutput p_2, TFOutput p_1) = Switch (pred, pred);
				TFOutput pivot_t = Identity (p_1, operName: "switch_t");
				TFOutput pivot_f = Identity (p_2, operName: "switch_f");
				pred = Identity (pred, operName: "pred_id");

				TFOutput res_t;
				TFOutput res_f;

				// Build the graph for the true branch in a new context.
				using (WithDependencies (pivot_t.Operation)) {
					res_t = true_fn ();
				}

				// Build the graph for the false branch in a new context.
				using (WithDependencies (pivot_f.Operation)) {
					res_f = false_fn ();
				}

				// Add the final merge to the graph.
				(TFOutput merges, TFOutput index) = Merge (new [] { res_f, res_t });

				return merges;
			}
		}

		/// <summary>
		///   Return elements from x or y depending on condition.
		/// </summary>
		/// 
		/// <param name="condition">LabeledTensor of type `bool`.</param>
		/// <param name="x">LabeledTensor for values where condition is true.</param>
		/// <param name="y">LabeledTensor for values where condition is false.</param>
		/// <param name="name">Optional op name.</param>
		/// 
		/// <returns>The labeled tensor with values according to condition.</returns>
		/// 
		public TFOutput Where (TFOutput condition, TFOutput? x, TFOutput? y, string name = null)
		{
			// https://github.com/tensorflow/tensorflow/blob/d4ce3b4681b3a550c095b2cd18a79494d1cc4039/tensorflow/python/ops/array_ops.py#L2342
			if (x == null && y == null)
				return this.Where (input: condition, operName: name);
			else if (x != null && y != null)
				return this.Select (condition: condition, t: x.Value, e: y.Value, operName: name);
			throw new ArgumentException ("x and y must both be non-None or both be None.");
		}

		/// <summary>
		/// Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.
		/// </summary>
		/// <remarks>
		///  Packs the list of tensors in <paramref name="values"/> into a tensor with rank one higher than
		///  each tensor in <paramref name="values"/>, by packing them along the <paramref name="axis"/> dimension.
		///  Given a list of length <c>N</c> of tensors of shape <c>(A, B, C)</c>: if <c>axis == 0</c> then the 
		///  <c>output</c> tensor will have the shape <c>(N, A, B, C)</c>; if <c>axis == 1</c> then the <c>output</c>
		///  tensor will have the shape <c>(A, N, B, C)</c>; etc.
		/// </remarks>
		/// 
		public TFOutput Stack (TFOutput [] values, int? axis = 0, string operName = "stack")
		{
			// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_ops.py#L804

			int ndims = GetTensorNumDims (values [0]);

			int expanded_num_dims = ndims + 1;
			if (axis < -expanded_num_dims || axis >= expanded_num_dims)
				throw new InvalidOperationException ($"axis = {axis} not in [{-expanded_num_dims}, {expanded_num_dims}]");

			return Pack (values, axis: axis, operName: operName);
		}

		/// <summary>
		/// Creates a sequence of numbers.
		/// </summary>
		/// <remarks>
		/// Creates a sequence of numbers that begins at `start` and extends by increments of `delta` up to but not including 
		/// `limit`. The dtype of the resulting tensor is inferred from the inputs unless it is provided explicitly.
		/// </remarks>
		/// <param name="start">A 0 - D `Tensor` (scalar).Acts as first entry in the range if `limit` is not None; otherwise, acts as range limit and first entry defaults to 0.</param>
		/// <param name="limit">A 0 - D `Tensor` (scalar).Upper limit of sequence, exclusive. If None, defaults to the value of `start` while the first entry of the range defaults to 0.</param>
		/// <param name="delta">A 0 - D `Tensor` (scalar).Number that increments `start`. Defaults to 1.</param>
		/// <param name="dataType">The type of the elements of the resulting tensor.</param>
		/// <param name="operName">A name for the operation.Defaults to "range".</param>
		public TFOutput Range (TFOutput start, TFOutput? limit = null, TFOutput? delta = null, TFDataType? dataType = null, string operName = "range")
		{
			// https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/math_ops.py#L1156

			if (limit == null) {
				limit = start;
				start = Cast (Const (new TFTensor (0.0)), start.OutputType); // TODO: Maybe add dataType as convenience in Const?
			}

			if (delta == null)
				delta = Cast (Const (new TFTensor (1.0)), start.OutputType);

			using (var newScope = WithScope (MakeName ("Range", operName))) {
				// infer dtype if not explicitly provided
				if (dataType == null) {
					var dtype_hierarchy = new [] { TFDataType.Int32, TFDataType.Int64, TFDataType.Float, TFDataType.Double };
					if (!dtype_hierarchy.Contains (start.OutputType)
					 || !dtype_hierarchy.Contains (limit.Value.OutputType)
					 || !dtype_hierarchy.Contains (delta.Value.OutputType))
						throw new ArgumentException ("Unexpected type");

					TFDataType [] dtypes = new [] { start.OutputType, limit.Value.OutputType, delta.Value.OutputType };
					int imax = dtypes.Select (x => Array.IndexOf (dtype_hierarchy, x)).Max ();
					TFDataType inferred_dtype = dtype_hierarchy [imax];

					start = Cast (start, inferred_dtype);
					limit = Cast (limit.Value, inferred_dtype);
					delta = Cast (delta.Value, inferred_dtype);
				}

				return Range (start, limit.Value, delta.Value, operName: operName);
			}
		}

	}
}